"""Phase 8 integrated pipeline benchmark."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase5.benchmark_long_context_real_corpus import read_text, tokenize_splits
from lsl import GenerationController, IntegratedLSLAgent


def evaluate(args):
    train_text, eval_text, corpus_path = read_text(args)
    tokenizer, train_tokens, eval_tokens = tokenize_splits(train_text, eval_text, args)
    train_text = tokenizer.decode(train_tokens[: args.max_train_tokens])
    eval_prompt = tokenizer.decode(eval_tokens[: args.prompt_tokens])

    agent = IntegratedLSLAgent(vocab_size=args.vocab_size, seed=args.seed)
    agent.build_tokenizer(train_text)
    chunks = [train_text[i:i + args.chunk_chars] for i in range(0, min(len(train_text), args.max_train_chars), args.chunk_chars)]
    t0 = time.perf_counter_ns()
    agent.observe_texts(chunks, source="integrated")
    observe_us = (time.perf_counter_ns() - t0) / 1000.0 / max(1, len(chunks))
    for i in range(args.items):
        agent.observe_text(
            f"The launch code for entity-{i:07d} is value_{i * 31 + 7}.",
            source=f"fact:{i}",
            learn_transitions=False,
        )
        agent.observe_event(f"node_{i}_0", "next", f"node_{i}_1", episode_id=i, evidence_id=i)
        agent.observe_event(f"node_{i}_1", "next", f"node_{i}_2", episode_id=i, evidence_id=i)

    qa_correct = 0
    event_correct = 0
    math_correct = 0
    latencies = []
    for i in range(args.items):
        q = f"What is the launch code for entity-{i:07d}?"
        t0 = time.perf_counter_ns()
        qa_correct += int(agent.answer(q) == f"value_{i * 31 + 7}")
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
        event_correct += int(agent.answer(f"Starting from node_{i}_0, follow next then next?") == f"node_{i}_2")
        math_correct += int(agent.answer(f"Start at {i % 13}. Add 3, multiply by 2.") == str((i % 13 + 3) * 2))
    generated = agent.generate(eval_prompt, max_new_tokens=args.generate_tokens)
    gen_metrics = GenerationController.generation_metrics(agent.tokenizer.encode(generated))
    diagnostics = agent.diagnostics()
    metrics = {
        "qa_accuracy": qa_correct / max(1, args.items),
        "event_accuracy": event_correct / max(1, args.items),
        "math_accuracy": math_correct / max(1, args.items),
        "generation_score": gen_metrics["coherence"],
        "loop_rate": gen_metrics["loop_rate"],
        "p50_latency_us": float(np.percentile(latencies, 50)),
        "observe_us_per_chunk": float(observe_us),
        "full_scan": diagnostics.get("last_full_scan", 0.0),
    }
    checks = {
        "qa": metrics["qa_accuracy"] >= args.accuracy_target,
        "events": metrics["event_accuracy"] >= args.accuracy_target,
        "math": metrics["math_accuracy"] >= args.accuracy_target,
        "generation": metrics["generation_score"] >= args.generation_target,
        "loop": metrics["loop_rate"] <= args.loop_target,
        "no_scan": metrics["full_scan"] == 0.0,
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "sample": {"prompt": eval_prompt[:160], "generated": generated[:320]},
        "corpus_path": corpus_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "tinystories_full", "wikitext2", "custom"], default="tinystories")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--wikitext-cache-dir", type=str, default=None)
    parser.add_argument("--tinystories-cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=2500)
    parser.add_argument("--tokenizer", choices=["word", "subword"], default="subword")
    parser.add_argument("--tokenizer-train-chars", type=int, default=120000)
    parser.add_argument("--max-train-chars", type=int, default=160000)
    parser.add_argument("--max-eval-chars", type=int, default=50000)
    parser.add_argument("--subword-max-merges", type=int, default=500)
    parser.add_argument("--subword-min-pair-count", type=int, default=3)
    parser.add_argument("--max-train-tokens", type=int, default=5000)
    parser.add_argument("--max-eval-tokens", type=int, default=1000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--chunk-chars", type=int, default=240)
    parser.add_argument("--items", type=int, default=96)
    parser.add_argument("--prompt-tokens", type=int, default=12)
    parser.add_argument("--generate-tokens", type=int, default=64)
    parser.add_argument("--accuracy-target", type=float, default=0.80)
    parser.add_argument("--generation-target", type=float, default=0.75)
    parser.add_argument("--loop-target", type=float, default=0.05)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    m = result["metrics"]
    ok = bool(result["success"])
    print("Phase 8: Integrated Agent")
    print("=" * 88)
    print(f"QA/event/math:      {m['qa_accuracy']:.2%} / {m['event_accuracy']:.2%} / {m['math_accuracy']:.2%}")
    print(f"Generation score:   {m['generation_score']:.3f}")
    print(f"Loop rate:          {m['loop_rate']:.3%}")
    print(f"p50 latency:        {m['p50_latency_us']:.3f} us")
    print(f"Overall status:     {'PASS' if ok else 'FAIL'}")
    print("-" * 88)
    print(f"Generated: {result['sample']['generated'][:280]}")
    payload = {"benchmark": "phase8_integrated_agent", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
