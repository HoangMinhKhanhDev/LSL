"""Phase 8 integrated scaling law smoke benchmark."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import IntegratedLSLAgent


def run_size(items: int, args):
    agent = IntegratedLSLAgent(vocab_size=1000, seed=args.seed)
    agent.build_tokenizer("launch code entity value")
    t0 = time.perf_counter_ns()
    for i in range(items):
        agent.observe_text(f"The launch code for entity-{i:07d} is value_{i}.", source=f"scale:{i}")
    train_us = (time.perf_counter_ns() - t0) / 1000.0 / max(1, items)
    correct = 0
    for i in range(items):
        correct += int(agent.answer(f"What is the launch code for entity-{i:07d}?") == f"value_{i}")
    accuracy = correct / max(1, items)
    proxy_loss = 1.0 - accuracy + 1.0 / max(1, items) ** 0.5
    return {
        "items": float(items),
        "accuracy": float(accuracy),
        "proxy_loss": float(proxy_loss),
        "train_us_per_item": float(train_us),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [run_size(size, args) for size in args.sizes]
    losses = [row["proxy_loss"] for row in rows]
    monotonic = all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))
    latency_ratio = max(row["train_us_per_item"] for row in rows) / max(1e-9, min(row["train_us_per_item"] for row in rows))
    checks = {
        "monotonic_loss": monotonic,
        "accuracy": all(row["accuracy"] >= 0.95 for row in rows),
        "latency_sublinear_proxy": latency_ratio <= 2.5,
    }
    ok = all(checks.values())
    print("Phase 8: External Scaling")
    print("=" * 88)
    for row in rows:
        print(f"{int(row['items']):>6} acc={row['accuracy']:.2%} loss={row['proxy_loss']:.4f} train_us={row['train_us_per_item']:.2f}")
    print(f"Latency ratio:  {latency_ratio:.2f}x")
    print(f"Overall status: {'PASS' if ok else 'FAIL'}")
    payload = {
        "benchmark": "phase8_external_scaling",
        "success": bool(ok),
        "checks": checks,
        "latency_ratio": float(latency_ratio),
        "rows": rows,
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
