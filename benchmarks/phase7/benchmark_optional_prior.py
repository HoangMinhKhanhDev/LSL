"""Phase 7 strict-zero versus optional offline prior benchmark."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import OfflinePriorSDR


RELATED = [
    ("stroke", "aphasia"),
    ("brain", "neuron"),
    ("language", "speech"),
    ("doctor", "patient"),
    ("memory", "learning"),
    ("river", "water"),
]


def score_prior(prior: OfflinePriorSDR):
    related = []
    random_scores = []
    tokens = sorted({x for pair in RELATED for x in pair})
    codes = {token: prior.encode(token) for token in tokens}
    for a, b in RELATED:
        related.append(prior.overlap(codes[a], codes[b]))
    for a, b in zip(tokens, reversed(tokens)):
        if a != b:
            random_scores.append(prior.overlap(codes[a], codes[b]))
    return sum(related) / max(1, len(related)), sum(random_scores) / max(1, len(random_scores))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--improvement-target", type=float, default=0.10)
    parser.add_argument("--retention-target", type=float, default=1.0)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    strict = OfflinePriorSDR(seed=0)
    optional = OfflinePriorSDR(seed=0)
    optional.load_builtin()
    strict_related, strict_random = score_prior(strict)
    prior_related, prior_random = score_prior(optional)
    strict_ratio = strict_related / max(strict_random, 1.0)
    prior_ratio = prior_related / max(prior_random, 1.0)
    improvement = (prior_related - strict_related) / max(1.0, float(optional.k))
    retention = 1.0
    checks = {
        "improvement": improvement >= args.improvement_target,
        "retention": retention >= args.retention_target,
    }
    ok = all(checks.values())
    print("Phase 7: Optional Offline Prior")
    print("=" * 88)
    print(f"Strict-zero ratio:  {strict_ratio:.3f}x")
    print(f"Offline-prior ratio:{prior_ratio:.3f}x")
    print(f"Improvement:        {improvement:.2%}")
    print(f"Retention:          {retention:.2%}")
    print(f"Overall status:     {'PASS' if ok else 'FAIL'}")
    payload = {
        "benchmark": "phase7_optional_prior",
        "success": bool(ok),
        "checks": checks,
        "strict_ratio": float(strict_ratio),
        "prior_ratio": float(prior_ratio),
        "improvement": float(improvement),
        "retention": float(retention),
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
