"""Phase 5 long-context memory capability benchmark."""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import LongContextMemory, LivingSynapseLM
from lsl.utils import softmax


def target_for_horizon(horizon: int) -> float:
    if horizon <= 1000:
        return 0.95
    if horizon <= 4000:
        return 0.90
    if horizon <= 16000:
        return 0.85
    if horizon <= 64000:
        return 0.75
    return 0.60


def build_fact_memory(horizon: int, args: argparse.Namespace) -> Tuple[LongContextMemory, Dict[Tuple[int, int], int]]:
    rng = np.random.default_rng(args.seed + horizon)
    memory = LongContextMemory(
        capacity=horizon + 4096,
        vocab_size=args.vocab_size,
        candidate_cap=args.candidate_cap,
        context_width=args.context_width,
        seed=args.seed,
    )
    expected = {}
    subjects = rng.choice(np.arange(10_000, 10_000 + horizon * 4), size=horizon, replace=False)
    relations = rng.choice(np.arange(1000, 1000 + args.relations), size=horizon, replace=True)
    objects = rng.choice(np.arange(1_000_000, 1_000_000 + horizon * 8), size=horizon, replace=False)
    for subject, relation, obj in zip(subjects, relations, objects):
        key = (int(subject), int(relation))
        expected[key] = int(obj)
        memory.observe_fact(int(subject), int(relation), int(obj))
    return memory, expected


def evaluate_fact_recall(horizon: int, args: argparse.Namespace) -> Dict[str, float]:
    memory, expected = build_fact_memory(horizon, args)
    rng = np.random.default_rng(args.seed + 99 + horizon)
    keys = list(expected.keys())
    sampled = rng.choice(len(keys), size=min(args.trials, len(keys)), replace=False)
    times = []
    candidates = []
    full_scans = 0
    correct = 0
    for idx in sampled:
        subject, relation = keys[int(idx)]
        t0 = time.perf_counter_ns()
        value = None
        for _ in range(args.lookup_repeats):
            value = memory.query_fact(subject, relation, allow_direct_lookup=False)
        times.append((time.perf_counter_ns() - t0) / 1000.0 / max(1, args.lookup_repeats))
        diag = memory.last_lookup_diag
        candidates.append(diag.get("candidate_count", 0.0))
        full_scans += int(diag.get("full_scan", 0.0))
        correct += int(value == expected[(subject, relation)])

    recall = correct / max(1, len(sampled))
    absent = 0
    for i in range(args.absent_queries):
        value = memory.query_fact(900_000_000 + i, 700_000 + i, allow_direct_lookup=False)
        score = memory.last_lookup_diag.get("best_score", 0.0)
        absent += int(value is not None and score >= args.min_match_bits)
    absent_fp = absent / max(1, args.absent_queries)

    return {
        "horizon": float(horizon),
        "fact_recall": float(recall),
        "target": float(target_for_horizon(horizon)),
        "p50_us": float(np.percentile(times, 50)),
        "p95_us": float(np.percentile(times, 95)),
        "candidate_p95": float(np.percentile(candidates, 95)),
        "full_scans": float(full_scans),
        "absent_false_positive": float(absent_fp),
        "success": float(
            recall >= target_for_horizon(horizon)
            and full_scans == 0
            and absent_fp <= args.absent_fp_target
            and np.percentile(candidates, 95) <= args.candidate_cap
        ),
    }


def evaluate_instruction(args: argparse.Namespace) -> Dict[str, float]:
    rng = np.random.default_rng(args.seed + 500)
    memory = LongContextMemory(capacity=args.instructions + 1024, vocab_size=args.vocab_size, seed=args.seed)
    commands = rng.choice(np.arange(40_000, 40_000 + args.instructions * 4), size=args.instructions, replace=False)
    responses = rng.choice(np.arange(80_000, 80_000 + args.instructions * 4), size=args.instructions, replace=False)
    expected = {}
    for command, response in zip(commands, responses):
        expected[int(command)] = int(response)
        memory.observe_instruction(int(command), int(response))

    correct = 0
    sampled = rng.choice(commands, size=min(args.trials, len(commands)), replace=False)
    for command in sampled:
        correct += int(memory.query_instruction(int(command), allow_direct_lookup=False) == expected[int(command)])
    return {"instruction_accuracy": float(correct / max(1, len(sampled)))}


def evaluate_next_token_gain(args: argparse.Namespace) -> Dict[str, float]:
    rng = np.random.default_rng(args.seed + 700)
    vocab = min(args.vocab_size, 256)
    motif_count = min(64, max(8, vocab // 4))
    motif_len = 6
    motifs = []
    for motif_id in range(motif_count):
        start = (motif_id * 7 + 3) % vocab
        motifs.append([
            start,
            (start + 17) % vocab,
            (start + 41) % vocab,
            (start + 73) % vocab,
            (start + 109) % vocab,
            (start + 149) % vocab,
        ])
    repeats = max(1, args.sequence_tokens // motif_len)
    schedule = rng.integers(0, motif_count, size=repeats)
    tokens = []
    for motif_id in schedule:
        tokens.extend(motifs[int(motif_id)])
    tokens = tokens[:args.sequence_tokens]

    plain = LivingSynapseLM(vocab_size=vocab, hidden_dim=64, use_sparse_computation=True, seed=args.seed)
    memory_model = LivingSynapseLM(
        vocab_size=vocab,
        hidden_dim=64,
        use_sparse_computation=True,
        use_long_context_memory=True,
        memory_candidate_cap=args.candidate_cap,
        long_context_strength=12.0,
        long_context_confidence_threshold=0.50,
        seed=args.seed,
    )
    split = int(0.7 * len(tokens))
    for i in range(split - 1):
        plain.observe(tokens[i], tokens[i + 1], store=True)
        memory_model.observe(tokens[i], tokens[i + 1], store=True)

    plain_loss = []
    memory_loss = []
    memory_top1 = 0
    total = 0
    start_eval = min(len(tokens) - 2, split + args.context_width)
    for i in range(start_eval, len(tokens) - 1):
        target = int(tokens[i + 1])
        p_plain = softmax(plain.forward(tokens[i]))
        p_memory = softmax(memory_model.forward(tokens[i]))
        plain_loss.append(-np.log(max(float(p_plain[target]), 1e-12)))
        memory_loss.append(-np.log(max(float(p_memory[target]), 1e-12)))
        memory_top1 += int(np.argmax(p_memory) == target)
        total += 1

    plain_mean = float(np.mean(plain_loss))
    memory_mean = float(np.mean(memory_loss))
    return {
        "plain_loss": plain_mean,
        "memory_loss": memory_mean,
        "loss_ratio": float(memory_mean / max(plain_mean, 1e-9)),
        "memory_top1": float(memory_top1 / max(1, total)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1000, 4000, 16000, 64000])
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--relations", type=int, default=32)
    parser.add_argument("--instructions", type=int, default=4096)
    parser.add_argument("--sequence-tokens", type=int, default=2500)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--lookup-repeats", type=int, default=8)
    parser.add_argument("--absent-queries", type=int, default=64)
    parser.add_argument("--absent-fp-target", type=float, default=0.01)
    parser.add_argument("--min-match-bits", type=int, default=10)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Phase 5: Long-Context Memory Capability")
    print("Sparse facts, instructions, and transition memory; no full-context scan.")

    facts = [evaluate_fact_recall(horizon, args) for horizon in sorted(args.horizons, reverse=True)]
    facts = sorted(facts, key=lambda row: row["horizon"])
    instruction = evaluate_instruction(args)
    next_token = evaluate_next_token_gain(args)

    latency_ratio = 1.0
    by_horizon = {int(row["horizon"]): row for row in facts}
    if min(by_horizon) in by_horizon and max(by_horizon) in by_horizon:
        latency_ratio = by_horizon[max(by_horizon)]["p50_us"] / max(by_horizon[min(by_horizon)]["p50_us"], 1e-9)

    ok = (
        all(bool(row["success"]) for row in facts)
        and latency_ratio <= 1.5
        and instruction["instruction_accuracy"] >= 0.90
        and next_token["loss_ratio"] <= 0.95
    )

    print("\n" + "=" * 96)
    print("LONG-CONTEXT MEMORY SUMMARY")
    print("=" * 96)
    print(f"{'Horizon':>10} {'FactRecall':>12} {'Target':>10} {'p50_us':>10} {'Cand95':>10} {'Status':>8}")
    print("-" * 96)
    for row in facts:
        print(
            f"{int(row['horizon']):>10,} "
            f"{100 * row['fact_recall']:>11.1f}% "
            f"{100 * row['target']:>9.1f}% "
            f"{row['p50_us']:>10.2f} "
            f"{row['candidate_p95']:>10.1f} "
            f"{'PASS' if row['success'] else 'FAIL':>8}"
        )
    print("-" * 96)
    print(f"Latency max/min:          {latency_ratio:.2f}x (target <=1.50x)")
    print(f"Instruction accuracy:     {instruction['instruction_accuracy']:.2%} (target >=90%)")
    print(f"Next-token loss ratio:    {next_token['loss_ratio']:.3f}x (target <=0.95x)")
    print(f"Memory model top1:        {next_token['memory_top1']:.2%}")
    print(f"Overall status:           {'PASS' if ok else 'FAIL'}")

    payload = {
        "benchmark": "long_context_memory",
        "success": bool(ok),
        "facts": facts,
        "instruction": instruction,
        "next_token": next_token,
        "latency_ratio": float(latency_ratio),
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
