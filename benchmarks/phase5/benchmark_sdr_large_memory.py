"""Phase 5 SDR large-memory retention and completion benchmark."""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class SparsePatternMemory:
    def __init__(self, sdr_dim: int, k: int, candidate_cap: int = 1024):
        self.sdr_dim = int(sdr_dim)
        self.k = int(k)
        self.candidate_cap = int(candidate_cap)
        self.patterns: List[Tuple[int, ...]] = []
        self.pattern_set = set()
        self.buckets = defaultdict(list)

    def observe(self, active: Tuple[int, ...]) -> None:
        pid = len(self.patterns)
        active = tuple(sorted(int(i) for i in active))
        self.patterns.append(active)
        self.pattern_set.add(active)
        for bit in active:
            self.buckets[bit].append(pid)

    def candidates(self, partial: Tuple[int, ...]) -> List[int]:
        seen = set()
        out = []
        for bit in partial:
            for pid in self.buckets.get(int(bit), []):
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(pid)
                if len(out) >= self.candidate_cap:
                    return out
        return out

    def complete(self, partial: Tuple[int, ...]) -> Tuple[int, ...]:
        partial_set = set(partial)
        best = ()
        best_score = -1
        for pid in self.candidates(partial):
            pat = self.patterns[pid]
            score = sum(1 for bit in pat if bit in partial_set)
            if score > best_score:
                best = pat
                best_score = score
        return best

    def recognize(self, active: Tuple[int, ...], threshold_ratio: float = 0.70) -> bool:
        completed = self.complete(active)
        if not completed:
            return False
        active_set = set(active)
        score = sum(1 for bit in completed if bit in active_set)
        return score >= int(round(self.k * threshold_ratio))


def generate_patterns(count: int, sdr_dim: int, k: int, seed: int) -> List[Tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    patterns = []
    seen = set()
    while len(patterns) < count:
        active = tuple(sorted(int(i) for i in rng.choice(sdr_dim, size=k, replace=False)))
        if active in seen:
            continue
        seen.add(active)
        patterns.append(active)
    return patterns


def overlap(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    return len(set(a) & set(b)) / max(1, len(a))


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    patterns = generate_patterns(args.patterns, args.sdr_dim, args.active_bits, args.seed)
    memory = SparsePatternMemory(args.sdr_dim, args.active_bits, args.candidate_cap)
    for pattern in patterns:
        memory.observe(pattern)

    rng = np.random.default_rng(args.seed + 10)
    sample_ids = rng.choice(len(patterns), size=min(args.samples, len(patterns)), replace=False)

    retention = []
    completion_50 = []
    completion_70 = []
    noisy = []
    for pid in sample_ids:
        pattern = patterns[int(pid)]
        retention.append(float(memory.complete(pattern) == pattern))
        active = list(pattern)
        rng.shuffle(active)
        partial_50 = tuple(sorted(active[: max(1, args.active_bits // 2)]))
        partial_70 = tuple(sorted(active[: max(1, int(args.active_bits * 0.30))]))
        completion_50.append(overlap(pattern, memory.complete(partial_50)))
        completion_70.append(overlap(pattern, memory.complete(partial_70)))
        noise = set(pattern)
        while len(noise) < args.active_bits + max(1, args.active_bits // 10):
            noise.add(int(rng.integers(args.sdr_dim)))
        noisy.append(float(memory.recognize(tuple(sorted(noise)))))

    false_hits = 0
    for _ in range(args.samples):
        random_pattern = tuple(sorted(int(i) for i in rng.choice(args.sdr_dim, size=args.active_bits, replace=False)))
        if random_pattern in memory.pattern_set:
            continue
        false_hits += int(memory.recognize(random_pattern))

    return {
        "patterns": float(args.patterns),
        "retention": float(np.mean(retention)),
        "false_positive": float(false_hits / max(1, args.samples)),
        "completion_50_mask": float(np.mean(completion_50)),
        "completion_70_mask": float(np.mean(completion_70)),
        "noisy_recognition": float(np.mean(noisy)),
        "candidate_cap": float(args.candidate_cap),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", type=int, default=100000)
    parser.add_argument("--sdr-dim", type=int, default=2048)
    parser.add_argument("--active-bits", type=int, default=20)
    parser.add_argument("--candidate-cap", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Phase 5: Interference-Free Storage + Pattern Completion")
    result = evaluate(args)
    checks = {
        "retention": result["retention"] >= 0.95,
        "false_positive": result["false_positive"] <= 0.01,
        "completion_50": result["completion_50_mask"] >= 0.90,
        "completion_70": result["completion_70_mask"] >= 0.75,
        "noisy": result["noisy_recognition"] >= 0.90,
    }
    ok = all(checks.values())

    print("\n" + "=" * 80)
    print("SDR LARGE MEMORY SUMMARY")
    print("=" * 80)
    print(f"Patterns stored:      {int(result['patterns']):,}")
    print(f"Retention:            {result['retention']:.2%} (target >=95%)")
    print(f"False positive:       {result['false_positive']:.2%} (target <=1%)")
    print(f"50% mask completion:  {result['completion_50_mask']:.2%} (target >=90%)")
    print(f"70% mask completion:  {result['completion_70_mask']:.2%} (target >=75%)")
    print(f"10% noisy recognition:{result['noisy_recognition']:.2%} (target >=90%)")
    print(f"Overall status:       {'PASS' if ok else 'FAIL'}")

    payload = {"benchmark": "sdr_large_memory", "success": bool(ok), "checks": checks, "metrics": result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
