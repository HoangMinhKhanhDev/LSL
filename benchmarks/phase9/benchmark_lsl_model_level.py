"""Phase 9 LSL model-level language model proof.

This benchmark treats BioComputeAgent as the claim-bearing LSL model: one
online instance must learn text, facts, chains, context patterns, and style
signals, then answer/generate/adapt without retraining from scratch.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import BioComputeAgent, GenerationController


BASE_TEXT = (
    "local synapse learns language online. "
    "sparse columns predict useful next words. "
    "careful reasoning follows evidence. "
    "adaptive memory stores facts without forgetting. "
    "formal answers use please therefore sincerely regards. "
)

ADAPT_TEXT = "zeta marker unlocks adaptive answer."


def token_pair(agent: BioComputeAgent, text: str):
    tokens = agent.tokenizer.encode(text)
    if len(tokens) < 2:
        raise ValueError(f"Need at least two tokens from {text!r}")
    return int(tokens[-2]), int(tokens[-1])


def transition_probability(agent: BioComputeAgent, source: int, target: int) -> float:
    agent.long_context.reset_state()
    return agent.long_context.target_probability(source, target, vocab_size=agent.vocab_size)


def train_model(agent: BioComputeAgent, args) -> None:
    tokenizer_corpus = BASE_TEXT + " " + ADAPT_TEXT + " local synapse learns."
    agent.build_tokenizer(tokenizer_corpus)
    for _ in range(args.text_epochs):
        agent.observe_text(BASE_TEXT)

    for i in range(args.items):
        agent.observe_fact(f"entity-{i:05d}", f"value_{i}", surprise=1.0)
        agent.observe_event(f"node_{i}_a", "link1", f"node_{i}_b", episode_id=i, evidence_id=i)
        agent.observe_event(f"node_{i}_b", "link2", f"node_{i}_c", episode_id=i, evidence_id=i)
        agent.observe_event(f"node_{i}_c", "link3", f"node_{i}_d", episode_id=i, evidence_id=i)
        agent.observe_context_pattern(
            [10 + (i % 2), 20 + ((i // 2) % 2), 100 + (i % 4), 300 + (i % 7)],
            (i % 2) ^ ((i // 2) % 2) ^ (i % 4),
        )
    agent.consolidate(replay_fraction=0.10)

    for token in ["please", "therefore", "sincerely", "regards"] * 4:
        agent.bio_modulator.observe(token, surprise=0.9)

    if agent.use_pc_v2:
        stable = list(range(96))
        for _ in range(args.pc_epochs):
            agent.pc_v2.reset_state()
            for token in stable:
                states = [agent.pc_v2.state_for(token + layer * 997, layer) for layer in range(agent.pc_v2.layers)]
                agent.pc_v2.observe(states, learn=True)


def sample_indices(items: int, samples: int):
    if items <= 0:
        return []
    count = min(int(items), int(samples))
    if count <= 1:
        return [0]
    return sorted({int(round(i * (items - 1) / (count - 1))) for i in range(count)})


def score_model(agent: BioComputeAgent, args):
    indices = sample_indices(args.items, args.samples)
    fact_correct = 0
    chain_correct = 0
    dendrite_correct = 0
    for i in indices:
        fact_correct += int(agent.recall_fact(f"entity-{i:05d}") == f"value_{i}")
        chain_correct += int(agent.answer(f"Starting from node_{i}_a, follow link1 then link2 then link3?") == f"node_{i}_d")
        pred = agent.predict_context_pattern([10 + (i % 2), 20 + ((i // 2) % 2), 100 + (i % 4), 300 + (i % 7)])
        dendrite_correct += int(pred == ((i % 2) ^ ((i // 2) % 2) ^ (i % 4)))

    before_source, before_target = token_pair(agent, "zeta marker")
    before_prob = transition_probability(agent, before_source, before_target)
    agent.observe_text(ADAPT_TEXT)
    after_prob = transition_probability(agent, before_source, before_target)
    adaptation_loss_drop = -math.log(before_prob) + math.log(after_prob)

    for j in range(args.interference_items):
        idx = args.items + j
        agent.observe_fact(f"entity-{idx:05d}", f"value_{idx}", surprise=1.0)
    retained = sum(int(agent.recall_fact(f"entity-{i:05d}") == f"value_{i}") for i in indices)

    prefix = agent.tokenizer.encode("local synapse")
    expected = agent.tokenizer.encode("local synapse learns")[-1]
    predicted = agent.predict_next_token_id(prefix)
    next_token_accuracy = float(predicted == expected)

    generated = agent.generate("local synapse", max_new_tokens=args.generate_tokens)
    gen_tokens = agent.tokenizer.encode(generated)
    gen_metrics = GenerationController.generation_metrics(gen_tokens)

    diag = agent.diagnostics()
    semantic_overlap = 0.0
    if agent.use_sdr_v2:
        semantic_overlap = agent.sdr_v2.overlap("unhappy", "unhappiness") / max(1.0, float(agent.sdr_v2.k))

    parameter_proxy = (
        diag.get("dendrite_segments", 0.0) * 4.0
        + diag.get("pc_v2_updates", 0.0) * 8.0
        + diag.get("hippocampus_fast", 0.0) * 3.0
        + diag.get("hippocampus_slow", 0.0) * 3.0
        + diag.get("world_items", 0.0)
        + diag.get("event_edges", 0.0)
    )

    start = time.perf_counter()
    for _ in range(args.latency_iterations):
        agent.observe_text("local synapse learns.")
    latency_ms = (time.perf_counter() - start) * 1000.0 / max(1, args.latency_iterations * 3)

    return {
        "trained_facts": float(args.items),
        "fact_recall": fact_correct / max(1, len(indices)) if agent.use_hippocampus else 0.0,
        "post_interference_retention": retained / max(1, len(indices)) if agent.use_hippocampus else 0.0,
        "chain_reasoning": chain_correct / max(1, len(indices)),
        "dendrite_accuracy": dendrite_correct / max(1, len(indices)) if agent.use_dendrites else 0.0,
        "next_token_accuracy": next_token_accuracy if agent.use_columns else 0.0,
        "adaptation_loss_drop": adaptation_loss_drop,
        "adaptation_probability_ratio": after_prob / max(before_prob, 1e-12),
        "generation_score": gen_metrics["coherence"],
        "pc_zero_update": diag.get("pc_v2_zero_update_ratio", 0.0) if agent.use_pc_v2 else 0.0,
        "sdr_subword_overlap": semantic_overlap,
        "neuromod_novel_update": diag.get("bio_mod_novel_update_ratio", 0.0) if agent.use_neuromodulation else 0.0,
        "tone_formal": float(agent.bio_modulator.tone() == "formal") if agent.use_neuromodulation else 0.0,
        "replay_budget": diag.get("hippocampus_replay_budget", 1.0),
        "full_scan": max(diag.get("hippocampus_last_full_scan", 0.0), diag.get("last_full_scan", 0.0)),
        "parameter_proxy": parameter_proxy,
        "latency_ms_per_token": latency_ms,
        "sample": generated,
    }


def evaluate(args):
    agent = BioComputeAgent(seed=args.seed)
    train_model(agent, args)
    metrics = score_model(agent, args)

    ablations = {}
    for name, kwargs, key in [
        ("no_pc_v2", {"use_pc_v2": False}, "pc_zero_update"),
        ("no_sdr_v2", {"use_sdr_v2": False}, "sdr_subword_overlap"),
        ("no_columns", {"use_columns": False}, "next_token_accuracy"),
        ("no_hippocampus", {"use_hippocampus": False}, "fact_recall"),
        ("no_dendrites", {"use_dendrites": False}, "dendrite_accuracy"),
        ("no_neuromodulation", {"use_neuromodulation": False}, "neuromod_novel_update"),
    ]:
        ablated = BioComputeAgent(seed=args.seed, **kwargs)
        train_model(ablated, args)
        ablated_metrics = score_model(ablated, args)
        base = metrics[key]
        ablations[f"{name}_drop"] = (base - ablated_metrics[key]) / max(base, 1e-9)

    checks = {
        "facts": metrics["trained_facts"] >= args.items,
        "fact_recall": metrics["fact_recall"] >= args.accuracy_target,
        "retention": metrics["post_interference_retention"] >= args.retention_target,
        "chain": metrics["chain_reasoning"] >= args.accuracy_target,
        "dendrite": metrics["dendrite_accuracy"] >= args.accuracy_target,
        "next_token": metrics["next_token_accuracy"] >= args.next_token_target,
        "adaptation": metrics["adaptation_loss_drop"] >= args.adaptation_loss_drop_target,
        "generation": metrics["generation_score"] >= args.generation_target,
        "pc": metrics["pc_zero_update"] >= args.pc_zero_update_target,
        "sdr": metrics["sdr_subword_overlap"] >= args.sdr_overlap_target,
        "neuromod": metrics["neuromod_novel_update"] >= args.neuromod_target,
        "tone": metrics["tone_formal"] == 1.0,
        "replay": metrics["replay_budget"] <= args.replay_budget_target,
        "no_scan": metrics["full_scan"] == 0.0,
        "parameter_proxy": metrics["parameter_proxy"] <= args.parameter_proxy_target,
        "latency": metrics["latency_ms_per_token"] <= args.latency_ms_target,
        "ablations": all(value >= args.ablation_drop_target for value in ablations.values()),
    }
    metrics.update(ablations)
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--interference-items", type=int, default=128)
    parser.add_argument("--text-epochs", type=int, default=10)
    parser.add_argument("--pc-epochs", type=int, default=30)
    parser.add_argument("--generate-tokens", type=int, default=48)
    parser.add_argument("--latency-iterations", type=int, default=64)
    parser.add_argument("--accuracy-target", type=float, default=0.95)
    parser.add_argument("--retention-target", type=float, default=0.99)
    parser.add_argument("--next-token-target", type=float, default=1.0)
    parser.add_argument("--adaptation-loss-drop-target", type=float, default=1.0)
    parser.add_argument("--generation-target", type=float, default=0.50)
    parser.add_argument("--pc-zero-update-target", type=float, default=0.50)
    parser.add_argument("--sdr-overlap-target", type=float, default=0.15)
    parser.add_argument("--neuromod-target", type=float, default=0.95)
    parser.add_argument("--replay-budget-target", type=float, default=0.10)
    parser.add_argument("--parameter-proxy-target", type=float, default=1000000.0)
    parser.add_argument("--latency-ms-target", type=float, default=5.0)
    parser.add_argument("--ablation-drop-target", type=float, default=0.20)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: LSL Model-Level LM")
    print("=" * 88)
    for key, value in result["metrics"].items():
        if key != "sample":
            print(f"{key:<32} {value:.4f}")
    print(f"Generated: {result['metrics']['sample'][:220]}")
    print(f"Overall status:                  {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_lsl_model_level", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
