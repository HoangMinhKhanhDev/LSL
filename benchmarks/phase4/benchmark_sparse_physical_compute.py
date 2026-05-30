"""Phase 4: sparse physical compute benchmark.

Mechanism under test:
  Sparse compute must be physically better than dense compute, not only lower
  in theoretical operation counts.

This benchmark measures wall-clock latency, peak Python allocation, touched
synapses, estimated memory traffic, cache/locality sensitivity, and an energy
proxy. Real joule/token is intentionally not claimed unless hardware energy
measurement is available.
"""
import argparse
import json
import os
import sys
import time
import tracemalloc
from typing import Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import LivingSynapseLayer, LivingSynapseLM, SDREncoder


FLOAT_BYTES = np.dtype(np.float32).itemsize
INTP_BYTES = np.dtype(np.intp).itemsize
CACHE_LINE_BYTES = 64


try:
    import psutil
except ImportError:
    psutil = None


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * float(pct)))
    return float(ordered[min(max(idx, 0), len(ordered) - 1)])


def latency_summary(times_us: List[float], total_seconds: float) -> Dict[str, float]:
    return {
        "median_us": percentile(times_us, 0.50),
        "p95_us": percentile(times_us, 0.95),
        "p99_us": percentile(times_us, 0.99),
        "ms_per_token": percentile(times_us, 0.50) / 1000.0,
        "tokens_per_sec": len(times_us) / max(total_seconds, 1e-12),
    }


def rss_bytes():
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def dense_layer_memory_bytes(dim: int) -> int:
    return int(3 * int(dim) * int(dim) * FLOAT_BYTES)


def dense_run_memory_bytes(dim: int) -> int:
    return int(5 * int(dim) * int(dim) * FLOAT_BYTES)


def active_state_bytes(active_count: int) -> int:
    return int(active_count) * INTP_BYTES


def dense_vector_from_active(dim: int, active: np.ndarray, values: np.ndarray) -> np.ndarray:
    x = np.zeros(int(dim), dtype=np.float32)
    x[active] = values
    return x


def make_sparse_vector(dim: int, active_count: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(int(dim), dtype=np.float32)
    active = rng.choice(int(dim), size=int(active_count), replace=False)
    x[active] = 1.0
    return x


def make_sparse_stream(
    dim: int,
    active_count: int,
    total: int,
    seed: int,
    stable: bool,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    if stable:
        return [make_sparse_vector(dim, active_count, rng)]
    return [make_sparse_vector(dim, active_count, rng) for _ in range(total)]


def random_active(dim: int, active_count: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(int(dim), size=int(active_count), replace=False)).astype(np.intp)


def clustered_active(dim: int, active_count: int, rng: np.random.Generator) -> np.ndarray:
    start = int(rng.integers(max(1, int(dim) - int(active_count) + 1)))
    return np.arange(start, start + int(active_count), dtype=np.intp)


def real_sdr_active(dim: int, active_count: int, total: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    enc = SDREncoder(dim, sparsity=float(active_count) / float(dim), seed=seed)
    result = []
    for _ in range(int(total)):
        code = enc.encode(rng.standard_normal(int(dim)).astype(np.float32))
        result.append(np.flatnonzero(code > 0.5).astype(np.intp))
    return result


def build_active_stream(
    dim: int,
    active_count: int,
    total: int,
    workload: str,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    if workload == "real_sdr":
        indices = real_sdr_active(dim, active_count, total, seed)
    elif workload == "clustered":
        indices = [clustered_active(dim, active_count, rng) for _ in range(total)]
    elif workload == "repeated":
        anchor = random_active(dim, active_count, rng)
        indices = [anchor for _ in range(total)]
    elif workload == "mixed":
        anchor = random_active(dim, active_count, rng)
        indices = []
        for i in range(total):
            if i % 7 == 0:
                indices.append(random_active(dim, active_count, rng))
            elif i % 3 == 0:
                indices.append(clustered_active(dim, active_count, rng))
            else:
                indices.append(anchor)
    else:
        indices = [random_active(dim, active_count, rng) for _ in range(total)]
    values = [np.ones(len(active), dtype=np.float32) for active in indices]
    return indices, values


def estimate_layer_bytes(stats: Dict[str, int], in_dim: int, out_dim: int) -> int:
    mode = stats.get("mode", "dense")
    if mode == "none" or int(stats.get("ops", 0)) <= 0:
        return 0
    active = max(1, int(stats.get("active_inputs", in_dim)))
    if mode in ("sparse", "sparse_active") or "sparse" in str(mode):
        touched = int(out_dim) * active
        weight_reads = 3 * touched
        fatigue_writes = touched
        temp_writes = touched
        vector_io = active + int(out_dim)
        return int((weight_reads + fatigue_writes + temp_writes + vector_io) * FLOAT_BYTES)
    touched = int(out_dim) * int(in_dim)
    weight_reads = 3 * touched
    fatigue_writes = touched
    effective_temp = touched
    signal_temp = touched
    vector_io = int(in_dim) + int(out_dim)
    return int((weight_reads + fatigue_writes + effective_temp + signal_temp + vector_io) * FLOAT_BYTES)


def estimate_update_bytes(stats: Dict[str, int], in_dim: int, out_dim: int) -> int:
    mode = stats.get("mode", "dense")
    if mode == "none" or int(stats.get("ops", 0)) <= 0:
        return 0
    active = max(1, int(stats.get("active_inputs", in_dim)))
    if "sparse" in str(mode):
        touched = int(out_dim) * active
        weight_rw = 2 * touched
        post_reads = int(out_dim) * active
        active_reads = active
        return int((weight_rw + post_reads + active_reads) * FLOAT_BYTES)
    touched = int(out_dim) * int(in_dim)
    weight_rw = 2 * touched
    outer_temp = touched
    vector_io = int(in_dim) + int(out_dim)
    return int((weight_rw + outer_temp + vector_io) * FLOAT_BYTES)


def layer_correctness(dim: int, active_count: int, seed: int) -> float:
    rng = np.random.default_rng(seed + 1000)
    active = random_active(dim, active_count, rng)
    values = np.ones(len(active), dtype=np.float32)
    x = dense_vector_from_active(dim, active, values)
    dense = LivingSynapseLayer(dim, dim, seed=seed)
    sparse = LivingSynapseLayer(dim, dim, seed=seed)
    y_dense = dense.forward(x, use_sparse=False)
    y_sparse = sparse.forward_active(active, values)
    return float(np.max(np.abs(y_dense - y_sparse)))


def measure_layer_active_stream(
    dim: int,
    active_stream: List[np.ndarray],
    value_stream: List[np.ndarray],
    sparse_active: bool,
    iterations: int,
    warmup: int,
    seed: int,
) -> Dict[str, float]:
    layer = LivingSynapseLayer(dim, dim, seed=seed + 17)
    dense_vectors = None
    if not sparse_active:
        dense_vectors = [
            dense_vector_from_active(dim, active, values)
            for active, values in zip(active_stream, value_stream)
        ]

    for i in range(int(warmup)):
        pos = i % len(active_stream)
        if sparse_active:
            layer.forward_active(active_stream[pos], value_stream[pos])
            layer.hebbian_update_active(1.0, lr=0.001, decay=0.001, max_norm=1e9)
        else:
            layer.forward(dense_vectors[pos], use_sparse=False)
            layer.hebbian_update_dense(1.0, lr=0.001, decay=0.001, max_norm=1e9)

    times_us: List[float] = []
    ops = 0
    forward_touched = 0
    update_touched = 0
    bytes_est = 0
    rss_before = rss_bytes()

    tracemalloc.start()
    t0_total = time.perf_counter()
    for i in range(int(iterations)):
        pos = (i + int(warmup)) % len(active_stream)
        t0 = time.perf_counter_ns()
        if sparse_active:
            layer.forward_active(active_stream[pos], value_stream[pos])
            forward_stats = dict(layer.last_forward_ops)
            layer.hebbian_update_active(1.0, lr=0.001, decay=0.001, max_norm=1e9)
        else:
            layer.forward(dense_vectors[pos], use_sparse=False)
            forward_stats = dict(layer.last_forward_ops)
            layer.hebbian_update_dense(1.0, lr=0.001, decay=0.001, max_norm=1e9)
        update_stats = dict(layer.last_update_ops)
        dt = time.perf_counter_ns() - t0
        times_us.append(float(dt) / 1000.0)
        ops += int(forward_stats["ops"]) + int(update_stats["ops"])
        forward_touched += int(forward_stats["fatigue_touched"])
        update_touched += int(update_stats["weights_touched"])
        bytes_est += estimate_layer_bytes(forward_stats, dim, dim)
        bytes_est += estimate_update_bytes(update_stats, dim, dim)
    total_seconds = time.perf_counter() - t0_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = rss_bytes()

    n = max(1, int(iterations))
    result = latency_summary(times_us, total_seconds)
    result.update({
        "ops_per_token": float(ops) / n,
        "forward_touched_per_token": float(forward_touched) / n,
        "update_touched_per_token": float(update_touched) / n,
        "touched_per_token": float(forward_touched + update_touched) / n,
        "bytes_per_token": float(bytes_est) / n,
        "cache_lines_per_token": float(bytes_est) / (n * CACHE_LINE_BYTES),
        "peak_kb": float(peak) / 1024.0,
        "alloc_kb_per_token": float(peak) / (1024.0 * n) if sparse_active else 0.0,
        "rss_delta_kb": 0.0 if rss_before is None or rss_after is None else float(rss_after - rss_before) / 1024.0,
    })
    return result


def measure_layer_stream(
    dim: int,
    active_count: int,
    use_sparse: bool,
    stable: bool,
    iterations: int,
    warmup: int,
    seed: int,
) -> Dict[str, float]:
    total = int(iterations) + int(warmup)
    vectors = make_sparse_stream(dim, active_count, total, seed=seed, stable=stable)
    layer = LivingSynapseLayer(dim, dim, seed=seed + 17)

    for i in range(int(warmup)):
        layer.forward(vectors[i % len(vectors)], use_sparse=use_sparse)

    times_us: List[float] = []
    ops = 0
    touched = 0
    bytes_est = 0

    tracemalloc.start()
    t0_total = time.perf_counter()
    for i in range(int(iterations)):
        x = vectors[(i + int(warmup)) % len(vectors)]
        t0 = time.perf_counter_ns()
        layer.forward(x, use_sparse=use_sparse)
        dt = time.perf_counter_ns() - t0
        times_us.append(float(dt) / 1000.0)
        stats = layer.last_forward_ops
        ops += int(stats["ops"])
        touched += int(stats["fatigue_touched"])
        bytes_est += estimate_layer_bytes(stats, dim, dim)
    total_seconds = time.perf_counter() - t0_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "median_us": percentile(times_us, 0.50),
        "p95_us": percentile(times_us, 0.95),
        "total_seconds": float(total_seconds),
        "ops_per_token": float(ops) / max(1, int(iterations)),
        "touched_per_token": float(touched) / max(1, int(iterations)),
        "bytes_per_token": float(bytes_est) / max(1, int(iterations)),
        "peak_kb": float(peak) / 1024.0,
    }


def pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1e-12)


def run_primitive_test(dim: int, active_count: int, workload: str, args: argparse.Namespace) -> Dict[str, float]:
    print(f"\n{'=' * 88}")
    print(f"Primitive sparse physical compute: d={dim:,}, k={active_count:,}, workload={workload}")
    print(f"{'=' * 88}")

    correctness = layer_correctness(dim, active_count, args.seed)
    total = int(args.iterations) + int(args.warmup)
    active_stream, value_stream = build_active_stream(
        dim, active_count, total, workload, args.seed + 11
    )
    dense = measure_layer_active_stream(
        dim, active_stream, value_stream, False, args.iterations, args.warmup, args.seed + 1
    )
    sparse = measure_layer_active_stream(
        dim, active_stream, value_stream, True, args.iterations, args.warmup, args.seed + 1
    )

    latency_speedup = safe_ratio(dense["ms_per_token"], sparse["ms_per_token"])
    tps_speedup = safe_ratio(sparse["tokens_per_sec"], dense["tokens_per_sec"])
    ops_speedup = safe_ratio(dense["ops_per_token"], sparse["ops_per_token"])
    touched_speedup = safe_ratio(dense["touched_per_token"], sparse["touched_per_token"])
    traffic_speedup = safe_ratio(dense["bytes_per_token"], sparse["bytes_per_token"])
    cache_speedup = safe_ratio(dense["cache_lines_per_token"], sparse["cache_lines_per_token"])
    peak_speedup = safe_ratio(dense["peak_kb"], sparse["peak_kb"])
    alloc_speedup = safe_ratio(dense["alloc_kb_per_token"], sparse["alloc_kb_per_token"])
    dense_state = int(dim) * FLOAT_BYTES
    sparse_state = active_state_bytes(active_count)
    active_state_ratio = safe_ratio(sparse_state, dense_state)
    joule_speedup = 0.0
    joule_status = "NOT_MEASURED"
    if args.dense_joule_per_token is not None and args.sparse_joule_per_token is not None:
        joule_speedup = safe_ratio(args.dense_joule_per_token, args.sparse_joule_per_token)
        joule_status = pass_fail(joule_speedup >= args.joule_target)

    ok = (
        correctness <= args.correctness_atol
        and latency_speedup >= args.latency_target
        and ops_speedup >= args.ops_target
        and traffic_speedup >= args.bytes_target
        and cache_speedup >= args.cache_target
        and touched_speedup >= args.bytes_target
        and peak_speedup >= args.ram_target
        and active_state_ratio <= args.active_state_ratio_target
        and sparse["alloc_kb_per_token"] <= args.sparse_alloc_kb_target
    )
    if args.require_real_energy:
        ok = ok and joule_status == "PASS"

    print(
        f"{'case':<12} {'p50_us':>10} {'p95_us':>10} {'p99_us':>10} {'tok/s':>10} "
        f"{'ops/token':>14} {'bytes/token':>14} {'cache_lines':>12} {'peak_kb':>10} {'allocKB/t':>10}"
    )
    print("-" * 88)
    for name, result in (
        ("dense", dense),
        ("sparse", sparse),
    ):
        print(
            f"{name:<12} {result['median_us']:>10.2f} {result['p95_us']:>10.2f} "
            f"{result['p99_us']:>10.2f} {result['tokens_per_sec']:>10.1f} "
            f"{result['ops_per_token']:>14,.0f} {result['bytes_per_token']:>14,.0f} "
            f"{result['cache_lines_per_token']:>12,.0f} {result['peak_kb']:>10.1f} "
            f"{result['alloc_kb_per_token']:>10.3f}"
        )
    print("-" * 88)
    print(f"Correctness max|dense-sparse|: {correctness:.3e}")
    print(f"Latency speedup:               {latency_speedup:.2f}x")
    print(f"Tokens/sec speedup:            {tps_speedup:.2f}x")
    print(f"Ops speedup:                   {ops_speedup:.2f}x")
    print(f"Touched-synapse speedup:       {touched_speedup:.2f}x")
    print(f"Bytes/energy proxy speedup:    {traffic_speedup:.2f}x")
    print(f"Cache-line proxy speedup:      {cache_speedup:.2f}x")
    print(f"RAM peak speedup:              {peak_speedup:.2f}x")
    print(f"Allocation speedup:            {alloc_speedup:.2f}x")
    print(f"Sparse active state ratio:     {100*active_state_ratio:.2f}%")
    print(f"Real joule/token:              {joule_status}")
    print(f"Primitive status:              {pass_fail(ok)}")

    return {
        "dim": float(dim),
        "active_count": float(active_count),
        "workload": workload,
        "correctness": correctness,
        "latency_speedup": latency_speedup,
        "tps_speedup": tps_speedup,
        "ops_speedup": ops_speedup,
        "touched_speedup": touched_speedup,
        "traffic_speedup": traffic_speedup,
        "cache_speedup": cache_speedup,
        "peak_speedup": peak_speedup,
        "alloc_speedup": alloc_speedup,
        "active_state_ratio": active_state_ratio,
        "sparse_alloc_kb_per_token": sparse["alloc_kb_per_token"],
        "joule_speedup": joule_speedup,
        "joule_status": joule_status,
        "success": float(ok),
    }


def log_loss_from_logits(logits: np.ndarray, target: int) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    m = float(np.max(logits))
    log_z = m + float(np.log(np.sum(np.exp(logits - m))))
    return float(log_z - logits[int(target)])


def build_model(vocab_size: int, dim: int, sparse: bool, args: argparse.Namespace) -> LivingSynapseLM:
    return LivingSynapseLM(
        vocab_size=vocab_size,
        hidden_dim=dim,
        k_ratio=0.4,
        seed=args.seed,
        slow_init=0.08,
        use_sdr=True,
        sdr_sparsity=args.sparsity,
        use_predictive_coding=not args.no_predictive_coding,
        theta=0.02 if not args.no_predictive_coding else 0.0,
        use_semantic_sdr=False,
        use_sparse_computation=sparse,
    )


def token_stream(vocab_size: int, total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(int(total), dtype=np.int32) % max(2, int(vocab_size))
    noise = rng.integers(0, max(2, int(vocab_size)), size=int(total), dtype=np.int32)
    choose_noise = (np.arange(int(total)) % 11) == 0
    base[choose_noise] = noise[choose_noise]
    return base


def iter_layers(model: LivingSynapseLM) -> Iterable[Tuple[str, LivingSynapseLayer]]:
    yield "embed", model.embed
    yield "recurrent", model.recurrent
    yield "output", model.output
    yield "W_emb_pred", model.W_emb_pred
    yield "W_ssm_pred", model.W_ssm_pred
    yield "W_rec_pred", model.W_rec_pred
    yield "ssm_B", model.ssm.B_proj
    yield "ssm_C", model.ssm.C_proj


def collect_model_step_stats(model: LivingSynapseLM) -> Dict[str, float]:
    ops = 0
    touched = 0
    bytes_est = 0
    dense_ops = 0
    dense_layers = 0
    sparse_layers = 0
    ssm_ops = 0
    for name, layer in iter_layers(model):
        stats = layer.last_forward_ops
        update_stats = layer.last_update_ops
        layer_ops = int(stats.get("ops", 0)) + int(update_stats.get("ops", 0))
        layer_touched = int(stats.get("fatigue_touched", 0)) + int(update_stats.get("weights_touched", 0))
        layer_bytes = estimate_layer_bytes(stats, layer.in_dim, layer.out_dim)
        layer_bytes += estimate_update_bytes(update_stats, layer.in_dim, layer.out_dim)
        ops += layer_ops
        touched += layer_touched
        bytes_est += layer_bytes
        if stats.get("mode") == "dense" or "dense" in update_stats.get("mode", ""):
            dense_layers += 1
            dense_ops += layer_ops
        elif stats.get("mode") in ("sparse", "sparse_active") or "sparse" in update_stats.get("mode", ""):
            sparse_layers += 1
        if name.startswith("ssm_"):
            ssm_ops += layer_ops
    return {
        "ops": float(ops),
        "touched": float(touched),
        "bytes_est": float(bytes_est),
        "dense_ops": float(dense_ops),
        "dense_layers": float(dense_layers),
        "sparse_layers": float(sparse_layers),
        "ssm_ops": float(ssm_ops),
    }


def measure_model_forward(
    sparse: bool,
    tokens: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model = build_model(args.model_vocab, args.model_dim, sparse, args)
    for i in range(args.model_warmup):
        model.forward(int(tokens[i]))

    times_us: List[float] = []
    losses: List[float] = []
    ops = touched = bytes_est = dense_ops = ssm_ops = 0.0
    dense_layers = sparse_layers = 0.0

    tracemalloc.start()
    t0_total = time.perf_counter()
    for i in range(args.model_iterations):
        pos = args.model_warmup + i
        token = int(tokens[pos])
        target = int(tokens[pos + 1])
        t0 = time.perf_counter_ns()
        logits = model.forward(token)
        dt = time.perf_counter_ns() - t0
        times_us.append(float(dt) / 1000.0)
        losses.append(log_loss_from_logits(logits, target))
        stats = collect_model_step_stats(model)
        ops += stats["ops"]
        touched += stats["touched"]
        bytes_est += stats["bytes_est"]
        dense_ops += stats["dense_ops"]
        ssm_ops += stats["ssm_ops"]
        dense_layers += stats["dense_layers"]
        sparse_layers += stats["sparse_layers"]
    total_seconds = time.perf_counter() - t0_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = max(1, int(args.model_iterations))
    return {
        "median_us": percentile(times_us, 0.50),
        "p95_us": percentile(times_us, 0.95),
        "total_seconds": float(total_seconds),
        "loss": float(np.mean(losses)),
        "ops_per_token": ops / n,
        "touched_per_token": touched / n,
        "bytes_per_token": bytes_est / n,
        "peak_kb": float(peak) / 1024.0,
        "dense_ops_fraction": dense_ops / max(ops, 1.0),
        "ssm_ops_fraction": ssm_ops / max(ops, 1.0),
        "dense_layers_per_token": dense_layers / n,
        "sparse_layers_per_token": sparse_layers / n,
    }


def measure_model_observe(
    sparse: bool,
    tokens: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model = build_model(args.model_vocab, args.model_dim, sparse, args)
    for i in range(args.model_warmup):
        model.observe(int(tokens[i]), int(tokens[i + 1]))

    times_us: List[float] = []
    p_targets: List[float] = []

    tracemalloc.start()
    t0_total = time.perf_counter()
    for i in range(args.model_iterations):
        pos = args.model_warmup + i
        t0 = time.perf_counter_ns()
        info = model.observe(int(tokens[pos]), int(tokens[pos + 1]))
        dt = time.perf_counter_ns() - t0
        times_us.append(float(dt) / 1000.0)
        p_targets.append(float(info["p_target"]))
    total_seconds = time.perf_counter() - t0_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "median_us": percentile(times_us, 0.50),
        "p95_us": percentile(times_us, 0.95),
        "total_seconds": float(total_seconds),
        "p_target": float(np.mean(p_targets)),
        "peak_kb": float(peak) / 1024.0,
    }


def run_model_test(args: argparse.Namespace) -> Dict[str, float]:
    print(f"\n{'=' * 88}")
    print(
        f"End-to-end token pipeline: dim={args.model_dim:,}, "
        f"vocab={args.model_vocab:,}, pc={not args.no_predictive_coding}"
    )
    print(f"{'=' * 88}")

    tokens = token_stream(
        args.model_vocab,
        args.model_iterations + args.model_warmup + 2,
        args.seed + 200,
    )
    dense_forward = measure_model_forward(False, tokens, args)
    sparse_forward = measure_model_forward(True, tokens, args)
    dense_observe = measure_model_observe(False, tokens, args)
    sparse_observe = measure_model_observe(True, tokens, args)

    forward_speedup = safe_ratio(dense_forward["median_us"], sparse_forward["median_us"])
    observe_speedup = safe_ratio(dense_observe["median_us"], sparse_observe["median_us"])
    ops_speedup = safe_ratio(dense_forward["ops_per_token"], sparse_forward["ops_per_token"])
    touched_speedup = safe_ratio(dense_forward["touched_per_token"], sparse_forward["touched_per_token"])
    traffic_speedup = safe_ratio(dense_forward["bytes_per_token"], sparse_forward["bytes_per_token"])
    loss_ratio = safe_ratio(sparse_forward["loss"], dense_forward["loss"])
    peak_ratio = safe_ratio(sparse_forward["peak_kb"], dense_forward["peak_kb"])

    ok = (
        forward_speedup >= args.end_to_end_latency_target
        and observe_speedup >= args.observe_latency_target
        and ops_speedup >= args.end_to_end_ops_target
        and touched_speedup >= args.bytes_target
        and traffic_speedup >= args.bytes_target
        and loss_ratio <= args.quality_ratio_target
        and peak_ratio <= args.ram_target
    )

    print(
        f"{'case':<16} {'forward_us':>12} {'observe_us':>12} {'loss':>10} "
        f"{'p_target':>10} {'ops/token':>14} {'bytes_est':>14} {'peak_kb':>12}"
    )
    print("-" * 88)
    print(
        f"{'dense':<16} {dense_forward['median_us']:>12.2f} {dense_observe['median_us']:>12.2f} "
        f"{dense_forward['loss']:>10.3f} {dense_observe['p_target']:>10.4f} "
        f"{dense_forward['ops_per_token']:>14,.0f} {dense_forward['bytes_per_token']:>14,.0f} "
        f"{dense_forward['peak_kb']:>12.1f}"
    )
    print(
        f"{'sparse':<16} {sparse_forward['median_us']:>12.2f} {sparse_observe['median_us']:>12.2f} "
        f"{sparse_forward['loss']:>10.3f} {sparse_observe['p_target']:>10.4f} "
        f"{sparse_forward['ops_per_token']:>14,.0f} {sparse_forward['bytes_per_token']:>14,.0f} "
        f"{sparse_forward['peak_kb']:>12.1f}"
    )
    print("-" * 88)
    print(f"Forward latency speedup:       {forward_speedup:.2f}x")
    print(f"Observe latency speedup:       {observe_speedup:.2f}x")
    print(f"Ops speedup:                   {ops_speedup:.2f}x")
    print(f"Touched-synapse speedup:       {touched_speedup:.2f}x")
    print(f"Energy proxy speedup:          {traffic_speedup:.2f}x")
    print(f"Sparse/dense loss ratio:       {loss_ratio:.2f}x")
    print(f"Sparse peak/dense peak:        {peak_ratio:.2f}x")
    print(f"Sparse dense-op fraction:      {sparse_forward['dense_ops_fraction']:.2f}")
    print(f"Sparse SSM-op fraction:        {sparse_forward['ssm_ops_fraction']:.2f}")
    print(f"Dense layers/token in sparse:  {sparse_forward['dense_layers_per_token']:.1f}")
    print(f"Sparse layers/token in sparse: {sparse_forward['sparse_layers_per_token']:.1f}")
    print(f"End-to-end status:             {pass_fail(ok)}")

    return {
        "forward_speedup": forward_speedup,
        "observe_speedup": observe_speedup,
        "ops_speedup": ops_speedup,
        "touched_speedup": touched_speedup,
        "traffic_speedup": traffic_speedup,
        "loss_ratio": loss_ratio,
        "peak_ratio": peak_ratio,
        "dense_ops_fraction": sparse_forward["dense_ops_fraction"],
        "ssm_ops_fraction": sparse_forward["ssm_ops_fraction"],
        "success": float(ok),
    }


def print_bottlenecks(primitive_results: List[Dict[str, float]], model_result: Dict[str, float], args: argparse.Namespace) -> None:
    notes: List[str] = []
    if any(r["latency_speedup"] < args.latency_target for r in primitive_results):
        notes.append("primitive sparse latency below strict latency target")
    if any(r["ops_speedup"] < args.ops_target for r in primitive_results):
        notes.append("primitive ops speedup below strict target")
    if any(r["traffic_speedup"] < args.bytes_target for r in primitive_results):
        notes.append("bytes/energy proxy speedup below strict target")
    if any(r["cache_speedup"] < args.cache_target for r in primitive_results):
        notes.append("cache-line proxy speedup below strict target")
    if any(r["peak_speedup"] < args.ram_target for r in primitive_results):
        notes.append("RAM peak speedup below strict target")
    if any(r["active_state_ratio"] > args.active_state_ratio_target for r in primitive_results):
        notes.append("sparse active state memory too high vs dense")
    if any(r["sparse_alloc_kb_per_token"] > args.sparse_alloc_kb_target for r in primitive_results):
        notes.append("sparse hot-path allocation too high")
    if model_result["forward_speedup"] < args.end_to_end_latency_target:
        notes.append("end-to-end forward latency below strict target")
    if model_result["observe_speedup"] < args.observe_latency_target:
        notes.append("online observe/adaptation latency below strict target")
    if model_result["ssm_ops_fraction"] > 0.30:
        notes.append("SSM dense bottleneck visible in sparse model path")
    if model_result["dense_ops_fraction"] > 0.25:
        notes.append("remaining dense layer operations dominate sparse path")
    if model_result["loss_ratio"] > args.quality_ratio_target:
        notes.append("sparse path quality sanity degraded beyond tolerance")

    print("\nBottleneck diagnosis:")
    if notes:
        for note in notes:
            print(f"  - {note}")
    else:
        print("  - no bottlenecks detected at configured thresholds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--workloads", nargs="+", default=["random", "clustered", "real_sdr", "repeated", "mixed"])
    parser.add_argument("--sparsity", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--model-vocab", type=int, default=96)
    parser.add_argument("--model-iterations", type=int, default=120)
    parser.add_argument("--model-warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-predictive-coding", action="store_true")
    parser.add_argument("--correctness-atol", type=float, default=1e-5)
    parser.add_argument("--latency-target", type=float, default=40.0)
    parser.add_argument("--end-to-end-latency-target", type=float, default=15.0)
    parser.add_argument("--observe-latency-target", type=float, default=5.0)
    parser.add_argument("--ops-target", type=float, default=40.0)
    parser.add_argument("--end-to-end-ops-target", type=float, default=25.0)
    parser.add_argument("--bytes-target", type=float, default=20.0)
    parser.add_argument("--cache-target", type=float, default=20.0)
    parser.add_argument("--ram-target", type=float, default=5.0)
    parser.add_argument("--active-state-ratio-target", type=float, default=0.05)
    parser.add_argument("--sparse-alloc-kb-target", type=float, default=2000.0)
    parser.add_argument("--quality-ratio-target", type=float, default=1.10)
    parser.add_argument("--dense-joule-per-token", type=float, default=None)
    parser.add_argument("--sparse-joule-per-token", type=float, default=None)
    parser.add_argument("--joule-target", type=float, default=10.0)
    parser.add_argument("--require-real-energy", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Phase 4: Sparse Physical Compute")
    print("Mechanism #2: CPU wall-clock, RAM, memory traffic, cache proxy, and energy proxy")
    print("Strict sparse primitive uses active_indices directly; no dense input scan in sparse hot path.")
    print("Real joule/token is NOT claimed unless --dense-joule-per-token and --sparse-joule-per-token are provided.")
    if psutil is None:
        print("RSS measurement: psutil unavailable; using tracemalloc peak allocation only.")

    primitive_results: List[Dict[str, float]] = []
    for dim in args.sizes:
        active_count = max(1, int(int(dim) * float(args.sparsity)))
        for workload in args.workloads:
            primitive_results.append(run_primitive_test(int(dim), active_count, workload, args))

    model_result = run_model_test(args)
    ok = all(bool(r["success"]) for r in primitive_results) and bool(model_result["success"])

    print(f"\n{'=' * 88}")
    print("SPARSE PHYSICAL COMPUTE SUMMARY")
    print(f"{'=' * 88}")
    print(
        f"{'scale/workload':<18} {'latency':>10} {'ops':>10} "
        f"{'bytes':>10} {'cache':>10} {'ram':>10} {'status':>10}"
    )
    print("-" * 88)
    for result in primitive_results:
        label = f"d={int(result['dim'])}/{result['workload']}"
        print(
            f"{label:<18} {result['latency_speedup']:>9.2f}x "
            f"{result['ops_speedup']:>9.2f}x {result['traffic_speedup']:>9.2f}x "
            f"{result['cache_speedup']:>9.2f}x {result['peak_speedup']:>9.2f}x "
            f"{pass_fail(bool(result['success'])):>10}"
        )
    print(
        f"{'end-to-end':<18} {model_result['forward_speedup']:>9.2f}x "
        f"{model_result['ops_speedup']:>9.2f}x {model_result['traffic_speedup']:>9.2f}x "
        f"{'n/a':>10} {model_result['peak_ratio']:>9.2f}x "
        f"{pass_fail(bool(model_result['success'])):>10}"
    )
    print("-" * 88)
    print_bottlenecks(primitive_results, model_result, args)
    print(f"\nMechanism #2 Sparse Physical Compute: {pass_fail(ok)}")
    payload = {
        "benchmark": "sparse_physical_compute",
        "success": bool(ok),
        "primitive_results": primitive_results,
        "model_result": model_result,
        "predictive_coding": not bool(args.no_predictive_coding),
    }
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
