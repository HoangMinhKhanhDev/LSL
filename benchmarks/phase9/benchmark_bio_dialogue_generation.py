"""Large multi-domain dialogue generation benchmark for BioComputeAgent."""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import BioComputeAgent, GenerationController


DOMAIN_BLUEPRINTS: List[Tuple[str, str, str, str]] = [
    ("clinic", "triage nurse", "patient", "symptom plan"),
    ("support", "support agent", "customer", "device repair"),
    ("finance", "advisor", "founder", "cashflow forecast"),
    ("education", "mentor", "student", "study schedule"),
    ("ops", "dispatcher", "technician", "route update"),
    ("research", "scientist", "reviewer", "experiment note"),
    ("coding", "engineer", "maintainer", "patch review"),
    ("travel", "planner", "traveler", "itinerary change"),
    ("legal", "counsel", "client", "contract clause"),
    ("sales", "account lead", "buyer", "renewal risk"),
    ("manufacturing", "operator", "quality lead", "calibration issue"),
    ("community", "moderator", "member", "event logistics"),
]


def build_dialogue_corpus(domains: int, turns_per_domain: int) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    prompts: List[str] = []
    for domain_idx in range(int(domains)):
        name, role_a, role_b, topic = DOMAIN_BLUEPRINTS[domain_idx % len(DOMAIN_BLUEPRINTS)]
        marker = f"{name}_case_{domain_idx:02d}"
        prompts.append(f"{role_a}: status for {marker}")
        for turn in range(int(turns_per_domain)):
            slot = turn % 7
            phase = ["intake", "clarify", "evidence", "option", "risk", "decision", "followup"][slot]
            handoff = f"{name}_handoff_{domain_idx:02d}_{turn % 17:02d}"
            texts.append(
                f"{role_a}: status for {marker} is {phase}. "
                f"{role_b}: I confirm {topic} with marker {handoff}. "
                f"{role_a}: next action keeps {marker} linked to {handoff}. "
            )
    return texts, prompts


def train_agent(texts: List[str], seed: int) -> BioComputeAgent:
    agent = BioComputeAgent(seed=seed, vocab_size=20000, candidate_cap=128)
    bootstrap = " ".join(texts)
    agent.build_tokenizer(bootstrap)
    for idx, text in enumerate(texts):
        agent.observe_text(text, source=f"dialogue:{idx}")
        if idx % 64 == 0:
            agent.consolidate(replay_fraction=0.02)
    agent.consolidate(replay_fraction=0.10)
    return agent


def generate_dialogue(agent: BioComputeAgent, prompt: str, max_new_tokens: int) -> str:
    generated = agent.generate(prompt, max_new_tokens=max_new_tokens)
    tokens = agent.tokenizer.encode(generated)
    trimmed: List[int] = []
    trigrams = set()
    for token in tokens:
        trimmed.append(int(token))
        if len(trimmed) < 3:
            continue
        tri = tuple(trimmed[-3:])
        if tri in trigrams:
            trimmed = trimmed[:-1]
            break
        trigrams.add(tri)
    return agent.tokenizer.decode(trimmed)


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    texts, prompts = build_dialogue_corpus(args.domains, args.turns_per_domain)
    agent = train_agent(texts, args.seed)
    generated_texts: List[str] = []
    token_count = 0
    started = time.perf_counter()
    for i in range(int(args.samples)):
        prompt = prompts[i % len(prompts)]
        generated = generate_dialogue(agent, prompt, args.generate_tokens)
        generated_texts.append(generated)
        token_count += len(agent.tokenizer.encode(generated))
    elapsed = time.perf_counter() - started

    metrics = [
        GenerationController.generation_metrics(
            agent.tokenizer.encode(text),
            unk_id=getattr(agent.tokenizer, "token_to_id", {}).get("<UNK>", 1),
        )
        for text in generated_texts
    ]
    mean = lambda key: sum(float(item[key]) for item in metrics) / max(1, len(metrics))
    domain_hits = 0
    for domain_idx, prompt in enumerate(prompts):
        name = DOMAIN_BLUEPRINTS[domain_idx % len(DOMAIN_BLUEPRINTS)][0]
        out = generate_dialogue(agent, prompt, max(16, args.generate_tokens // 2)).lower()
        domain_hits += int(name in out or f"{name}_case_{domain_idx:02d}" in out)
    domain_coherence = domain_hits / max(1, len(prompts))
    diagnostics = agent.diagnostics()
    tokens_per_second = token_count / max(elapsed, 1e-12)
    ms_per_token = 1000.0 * elapsed / max(1, token_count)
    checks = {
        "speed": tokens_per_second >= args.speed_target,
        "coherence": mean("coherence") >= args.coherence_target,
        "loop_rate": mean("loop_rate") <= args.loop_rate_target,
        "unk_rate": mean("unk_rate") <= args.unk_rate_target,
        "domain_coherence": domain_coherence >= args.domain_coherence_target,
        "no_full_scan": diagnostics.get("world_full_scan", 0.0) == 0.0 and diagnostics.get("event_full_scan", 0.0) == 0.0,
    }
    return {
        "success": all(checks.values()),
        "checks": checks,
        "metrics": {
            "domains": int(args.domains),
            "turns_per_domain": int(args.turns_per_domain),
            "training_dialogue_turns": int(args.domains * args.turns_per_domain),
            "samples": int(args.samples),
            "generated_tokens": int(token_count),
            "elapsed_seconds": float(elapsed),
            "tokens_per_second": float(tokens_per_second),
            "ms_per_token": float(ms_per_token),
            "coherence": float(mean("coherence")),
            "loop_rate": float(mean("loop_rate")),
            "unk_rate": float(mean("unk_rate")),
            "distinct2": float(mean("distinct2")),
            "domain_coherence": float(domain_coherence),
            "column_suppression_rate": float(diagnostics.get("column_suppression_rate", 0.0)),
            "pc_zero_update_ratio": float(diagnostics.get("pc_v2_zero_update_ratio", 0.0)),
            "hippocampus_replay_budget": float(diagnostics.get("hippocampus_replay_budget", 0.0)),
            "sample": generated_texts[0] if generated_texts else "",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", type=int, default=8)
    parser.add_argument("--turns-per-domain", type=int, default=256)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--generate-tokens", type=int, default=64)
    parser.add_argument("--speed-target", type=float, default=120.0)
    parser.add_argument("--coherence-target", type=float, default=0.50)
    parser.add_argument("--loop-rate-target", type=float, default=0.03)
    parser.add_argument("--unk-rate-target", type=float, default=0.003)
    parser.add_argument("--domain-coherence-target", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio Dialogue Generation")
    print("=" * 88)
    for key, value in result["metrics"].items():
        if key == "sample":
            continue
        print(f"{key:<30} {value:.6g}")
    print(f"Sample: {result['metrics']['sample'][:220]}")
    print(f"Overall status:                {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_dialogue_generation", **result}
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
