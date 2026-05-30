"""Measure real energy for the native sparse LSL kernel and write evidence JSON."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.energy.power_sensors import PowerSensorUnavailable, create_power_sensor
from lsl import LivingSynapseLayer, NATIVE_AVAILABLE, require_native


def _dense_workload(layer: LivingSynapseLayer, dense_x: np.ndarray, tokens: int) -> None:
    for _ in range(int(tokens)):
        layer.forward(dense_x, use_sparse=False)
        layer.hebbian_update_dense(0.05, lr=0.00001, decay=0.00001, max_norm=12.0)


def _sparse_workload(layer: LivingSynapseLayer, active: np.ndarray, values: np.ndarray, tokens: int) -> None:
    for _ in range(int(tokens)):
        layer.forward_active(active, values)
        layer.hebbian_update_active(0.05, lr=0.00001, decay=0.00001, max_norm=12.0)


def run_measurement(args: argparse.Namespace) -> Dict[str, object]:
    require_native()
    rng = np.random.default_rng(args.seed)
    dense_layer = LivingSynapseLayer(args.dim, args.dim, seed=args.seed)
    sparse_layer = LivingSynapseLayer(args.dim, args.dim, seed=args.seed)
    dense_x = rng.standard_normal(args.dim).astype(np.float32)
    active = np.asarray(sorted(rng.choice(args.dim, size=args.active, replace=False)), dtype=np.intp)
    values = rng.uniform(0.5, 1.0, size=args.active).astype(np.float32)

    _dense_workload(dense_layer, dense_x, args.warmup)
    _sparse_workload(sparse_layer, active, values, args.warmup)

    sensor = create_power_sensor(args.sensor, dll_path=args.intel_power_gadget_dll)
    try:
        dense = sensor.measure(lambda: _dense_workload(dense_layer, dense_x, args.tokens))
        cooldown_until = time.perf_counter() + max(0.0, float(args.cooldown_seconds))
        while time.perf_counter() < cooldown_until:
            time.sleep(min(0.05, cooldown_until - time.perf_counter()))
        sparse = sensor.measure(lambda: _sparse_workload(sparse_layer, active, values, args.tokens))
    finally:
        sensor.close()

    dense_jpt = dense.joules / max(1, int(args.tokens))
    sparse_jpt = sparse.joules / max(1, int(args.tokens))
    savings = 1.0 - sparse_jpt / max(dense_jpt, 1e-12)
    evidence = {
        "method": sparse.method,
        "device": sparse.device or platform.processor() or platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokens": int(args.tokens),
        "dense_watts": float(dense.watts),
        "sparse_watts": float(sparse.watts),
        "dense_joule_per_token": float(dense_jpt),
        "sparse_joule_per_token": float(sparse_jpt),
        "savings": float(savings),
        "sensor": sparse.sensor,
        "native_available": bool(NATIVE_AVAILABLE),
        "dim": int(args.dim),
        "active": int(args.active),
        "dense_elapsed_seconds": float(dense.elapsed_seconds),
        "sparse_elapsed_seconds": float(sparse.elapsed_seconds),
        "dense_total_joules": float(dense.joules),
        "sparse_total_joules": float(sparse.joules),
        "dense_ops_per_token": int(args.dim * args.dim),
        "sparse_ops_per_token": int(args.dim * args.active),
        "sparse_kernel_mode": sparse_layer.last_forward_ops.get("mode", ""),
        "sparse_update_mode": sparse_layer.last_update_ops.get("mode", ""),
    }
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=os.path.join("results", "energy_evidence.json"))
    parser.add_argument("--sensor", choices=["auto", "intel_power_gadget", "windows_power_meter"], default="auto")
    parser.add_argument("--intel-power-gadget-dll", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--cooldown-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evidence = run_measurement(args)
    except PowerSensorUnavailable as exc:
        print(f"ENERGY SENSOR UNAVAILABLE: {exc}", file=sys.stderr)
        print(
            "Install Intel Power Gadget / expose EnergyLib64.dll, or enable a Windows Power Meter "
            "performance counter, then rerun this command.",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        print(f"NATIVE SPARSE KERNEL UNAVAILABLE: {exc}", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    print(f"Wrote real energy evidence: {args.output}")
    print(f"sparse_watts={evidence['sparse_watts']:.6f}")
    print(f"dense_joule_per_token={evidence['dense_joule_per_token']:.9f}")
    print(f"sparse_joule_per_token={evidence['sparse_joule_per_token']:.9f}")
    print(f"savings={100.0 * evidence['savings']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

