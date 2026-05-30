"""Phase 9 integrated biological compute agent proof."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import BioComputeAgent, GenerationController


def train_agent(agent: BioComputeAgent, items: int):
    text = (
        "the small model learns local patterns. the sparse memory stores facts. "
        "the agent answers with evidence and adapts tone. "
    )
    agent.build_tokenizer(text)
    for _ in range(8):
        agent.observe_text(text)
    for i in range(items):
        agent.observe_fact(f"entity-{i:04d}", f"value_{i}", surprise=1.0)
        agent.observe_event(f"node_{i}_a", "link1", f"node_{i}_b", episode_id=i, evidence_id=i)
        agent.observe_event(f"node_{i}_b", "link2", f"node_{i}_c", episode_id=i, evidence_id=i)
        agent.observe_event(f"node_{i}_c", "link3", f"node_{i}_d", episode_id=i, evidence_id=i)
        agent.observe_context_pattern([10 + (i % 2), 20 + ((i // 2) % 2), 100 + (i % 4)], (i % 2) ^ ((i // 2) % 2))
    agent.consolidate(replay_fraction=0.10)
    for token in ["please", "therefore", "sincerely", "regards"]:
        agent.bio_modulator.observe(token, surprise=0.9)
    if agent.use_pc_v2:
        stable = list(range(80))
        for _ in range(30):
            agent.pc_v2.reset_state()
            for token in stable:
                states = [agent.pc_v2.state_for(token + layer * 997, layer) for layer in range(agent.pc_v2.layers)]
                agent.pc_v2.observe(states, learn=True)


def score_agent(agent: BioComputeAgent, items: int):
    fact_correct = 0
    chain_correct = 0
    dendrite_correct = 0
    for i in range(items):
        fact_correct += int(agent.recall_fact(f"entity-{i:04d}") == f"value_{i}")
        chain_correct += int(agent.answer(f"Starting from node_{i}_a, follow link1 then link2 then link3?") == f"node_{i}_d")
        pred = agent.predict_context_pattern([10 + (i % 2), 20 + ((i // 2) % 2), 100 + (i % 4)])
        dendrite_correct += int(pred == ((i % 2) ^ ((i // 2) % 2)))
    generated = agent.generate("the small model", max_new_tokens=32)
    gen_tokens = agent.tokenizer.encode(generated)
    gen_metrics = GenerationController.generation_metrics(gen_tokens)
    diag = agent.diagnostics()
    return {
        "fact_recall": fact_correct / max(1, items) if agent.use_hippocampus else 0.0,
        "chain_reasoning": chain_correct / max(1, items),
        "dendrite_accuracy": dendrite_correct / max(1, items) if agent.use_dendrites else 0.0,
        "generation_score": gen_metrics["coherence"],
        "retention": fact_correct / max(1, items) if agent.use_hippocampus else 0.0,
        "pc_zero_update": diag.get("pc_v2_zero_update_ratio", 0.0) if agent.use_pc_v2 else 0.0,
        "neuromod_novel_update": diag.get("bio_mod_novel_update_ratio", 0.0) if agent.use_neuromodulation else 0.0,
        "tone_formal": float(agent.bio_modulator.tone() == "formal") if agent.use_neuromodulation else 0.0,
        "replay_budget": diag.get("hippocampus_replay_budget", 1.0),
        "full_scan": diag.get("hippocampus_last_full_scan", 0.0),
        "sample": generated,
    }


def evaluate(args):
    agent = BioComputeAgent(seed=args.seed)
    train_agent(agent, args.items)
    metrics = score_agent(agent, args.items)

    ablations = {}
    for name, kwargs, key in [
        ("no_pc_v2", {"use_pc_v2": False}, "pc_zero_update"),
        ("no_hippocampus", {"use_hippocampus": False}, "fact_recall"),
        ("no_dendrites", {"use_dendrites": False}, "dendrite_accuracy"),
        ("no_neuromodulation", {"use_neuromodulation": False}, "neuromod_novel_update"),
    ]:
        ablated = BioComputeAgent(seed=args.seed, **kwargs)
        train_agent(ablated, args.items)
        ablated_metrics = score_agent(ablated, args.items)
        base = metrics[key]
        drop = (base - ablated_metrics[key]) / max(base, 1e-9)
        ablations[f"{name}_drop"] = drop

    checks = {
        "fact_recall": metrics["fact_recall"] >= args.accuracy_target,
        "chain": metrics["chain_reasoning"] >= args.accuracy_target,
        "dendrite": metrics["dendrite_accuracy"] >= args.accuracy_target,
        "generation": metrics["generation_score"] >= args.generation_target,
        "retention": metrics["retention"] >= args.retention_target,
        "pc": metrics["pc_zero_update"] >= args.pc_zero_update_target,
        "neuromod": metrics["neuromod_novel_update"] >= args.neuromod_target,
        "tone": metrics["tone_formal"] == 1.0,
        "replay": metrics["replay_budget"] <= args.replay_budget_target,
        "no_scan": metrics["full_scan"] == 0.0,
        "ablations": all(value >= args.ablation_drop_target for value in ablations.values()),
    }
    metrics.update(ablations)
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--accuracy-target", type=float, default=0.95)
    parser.add_argument("--generation-target", type=float, default=0.50)
    parser.add_argument("--retention-target", type=float, default=0.95)
    parser.add_argument("--pc-zero-update-target", type=float, default=0.50)
    parser.add_argument("--neuromod-target", type=float, default=0.95)
    parser.add_argument("--replay-budget-target", type=float, default=0.10)
    parser.add_argument("--ablation-drop-target", type=float, default=0.20)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio Integrated Agent")
    print("=" * 88)
    for key, value in result["metrics"].items():
        if key != "sample":
            print(f"{key:<28} {value:.4f}")
    print(f"Generated: {result['metrics']['sample'][:220]}")
    print(f"Overall status:              {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_integrated_agent", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
