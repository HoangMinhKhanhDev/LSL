"""Benchmark native sparse kernels, tokenizer hot path, and lookup paths."""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import DatasetLoader, DendriticLayer, LSLCoreModel, SimpleSubwordTokenizer, write_result  # noqa: E402
from lsl.memory import SparseKeyValueMemory  # noqa: E402
from lsl.cortical_column import CorticalColumnSequenceMemory  # noqa: E402
from lsl import sparse_native  # noqa: E402


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def regex_tokens(text: str) -> List[str]:
    return [m.group(0) for m in re.finditer(r"\w+|[^\w\s]", str(text).lower(), re.UNICODE)]


def timed(fn):
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    return value, elapsed


def bytes_to_gb_per_s(num_bytes: float, seconds: float) -> float:
    return float(num_bytes) / max(1e-12, float(seconds)) / 1e9


def benchmark_tokenizer(text: str, vocab_size: int, max_tokens: int) -> Dict[str, object]:
    tokenizer = SimpleSubwordTokenizer(
        vocab_size=vocab_size,
        vietnamese_normalization=True,
        byte_fallback=True,
        cache_dir=os.path.join("results", "kernel_cache"),
    )
    tokenizer.build_vocab(text)
    tokenizer._encode_text_cache.clear()
    tokenizer._decode_cache.clear()
    native_tokens, native_seconds = timed(lambda: sparse_native.simple_tokenize(text, max_tokens=max_tokens))
    regex_only, regex_seconds = timed(lambda: regex_tokens(text)[:max_tokens])
    ids, encode_seconds = timed(lambda: tokenizer.encode(text, max_tokens=max_tokens))
    _, decode_seconds = timed(lambda: tokenizer.decode(ids))
    cached_ids, cached_encode_seconds = timed(lambda: tokenizer.encode(text, max_tokens=max_tokens))
    native_bytes = len(text.encode("utf-8"))
    cache = tokenizer.cache_stats()
    return {
        "mechanism": "tokenizer_hot_path",
        "tokenizer": "subword_vi",
        "tokens": len(ids),
        "native_tokenize_tps": len(native_tokens) / max(native_seconds, 1e-12),
        "regex_tokenize_tps": len(regex_only) / max(regex_seconds, 1e-12),
        "encode_tps": len(ids) / max(encode_seconds, 1e-12),
        "cached_encode_tps": len(cached_ids) / max(cached_encode_seconds, 1e-12),
        "decode_tps": len(ids) / max(decode_seconds, 1e-12),
        "unk_rate": float(tokenizer.unk_rate(ids)),
        "cache_hit_rate": float(cache.get("hit_rate", 0.0)),
        "bytes": native_bytes,
        "bandwidth_gbps": bytes_to_gb_per_s(native_bytes, encode_seconds),
        "cache_miss_proxy": float(1.0 - cache.get("hit_rate", 0.0)),
    }


def _build_lookup_memory(capacity: int, sdr_dim: int, sparsity: float, items: int) -> SparseKeyValueMemory:
    memory = SparseKeyValueMemory(capacity=capacity, sdr_dim=sdr_dim, sparsity=sparsity, candidate_cap=64)
    for i in range(items):
        memory.add(i, (i * 7) % 10007, vocab_size=50000)
    return memory


def benchmark_hippocampus_lookup(items: int, lookups: int) -> Dict[str, object]:
    memory = _build_lookup_memory(capacity=max(256, items), sdr_dim=4096, sparsity=0.02, items=items)
    queries = [(i * 17 + 5) % (items * 3) for i in range(lookups)]
    python_results, python_seconds = timed(
        lambda: [memory.lookup(q, vocab_size=50000, allow_direct_lookup=False, prefer_native_scoring=False) for q in queries]
    )
    native_results, native_seconds = timed(
        lambda: [memory.lookup(q, vocab_size=50000, allow_direct_lookup=False, prefer_native_scoring=True) for q in queries]
    )
    diag = memory.diagnostics()
    bytes_touched = float(diag["similarity_ops"]) * 4.0
    mismatches = sum(int(a != b) for a, b in zip(native_results, python_results))
    return {
        "mechanism": "hippocampus_lookup",
        "native_tps": float(lookups) / max(native_seconds, 1e-12),
        "python_tps": float(lookups) / max(python_seconds, 1e-12),
        "speedup": float(python_seconds / max(native_seconds, 1e-12)),
        "agreement": float(1.0 - mismatches / max(1, lookups)),
        "candidate_count": float(diag["candidate_count"]),
        "similarity_ops": float(diag["similarity_ops"]),
        "cache_miss_proxy": float(diag["candidate_count"]) / max(1.0, float(diag["sdr_dim"])),
        "bytes": bytes_touched,
        "bandwidth_gbps": bytes_to_gb_per_s(bytes_touched, native_seconds),
        "best_score": float(diag["best_score"]),
        "native_mismatch_count": float(mismatches),
        "native_lookup_available": float(diag.get("native_lookup_available", 0.0)),
    }


def benchmark_sparse_batch(batch_size: int, out_dim: int, in_dim: int, active_count: int, repeats: int) -> Dict[str, object]:
    rng = np.random.default_rng(0)
    slow = rng.random((out_dim, in_dim), dtype=np.float32)
    live = rng.random((out_dim, in_dim), dtype=np.float32)
    fatigue = np.zeros((out_dim, in_dim), dtype=np.float32)
    active = np.stack([np.sort(rng.choice(in_dim, size=active_count, replace=False)).astype(np.intp) for _ in range(batch_size)])
    values = np.ones((batch_size, active_count), dtype=np.float32)
    lengths = np.full(batch_size, active_count, dtype=np.intp)

    def run_native():
        return sparse_native.forward_active_batch(slow, live, fatigue.copy(), active, values, lengths)

    def run_single():
        outputs = []
        for row in range(batch_size):
            post, _ = sparse_native.forward_active(
                slow,
                live,
                fatigue.copy(),
                active[row],
                values[row],
            )
            outputs.append(post)
        return np.stack(outputs, axis=0)

    native_runs, native_seconds = timed(lambda: [run_native() for _ in range(repeats)])
    single_runs, python_seconds = timed(lambda: [run_single() for _ in range(repeats)])
    touched_bytes = float(repeats * batch_size * out_dim * active_count * 4 * 4)
    native_posts = native_runs[-1][0] if native_runs else np.zeros((batch_size, out_dim), dtype=np.float32)
    single_posts = single_runs[-1] if single_runs else np.zeros((batch_size, out_dim), dtype=np.float32)
    max_diff = float(np.max(np.abs(native_posts - single_posts))) if native_posts.size and single_posts.size else 0.0
    return {
        "mechanism": "sparse_batch_kernel",
        "native_tps": float(repeats * batch_size) / max(native_seconds, 1e-12),
        "python_tps": float(repeats * batch_size) / max(python_seconds, 1e-12),
        "speedup": float(python_seconds / max(native_seconds, 1e-12)),
        "max_abs_diff": max_diff,
        "cache_miss_proxy": float(active_count) / max(1.0, float(in_dim)),
        "bytes": touched_bytes,
        "bandwidth_gbps": bytes_to_gb_per_s(touched_bytes, native_seconds),
    }


def benchmark_dendrite(branches: int, active_bits: int, outputs: int, repeats: int) -> Dict[str, object]:
    layer = DendriticLayer(input_dim=4096, outputs=outputs, segment_size=4)
    for i in range(branches):
        bits = [((i * 13) + j * 17) % 4096 for j in range(4)]
        layer.add_branch(bits, output=i % outputs, threshold=3.2, weights=(1.0, 1.0, 1.0, 1.0))
    queries = [sorted({(i * 19 + j * 31) % 4096 for j in range(active_bits)}) for i in range(repeats)]

    native_preds, native_seconds = timed(lambda: [layer.predict(bits, prefer_native=True) for bits in queries])
    python_preds, python_seconds = timed(lambda: [layer.predict(bits, prefer_native=False) for bits in queries])
    diag = layer.diagnostics()
    touched_bytes = float(diag["last_ops"]) * 4.0
    return {
        "mechanism": "dendrite_predict",
        "native_tps": float(repeats) / max(native_seconds, 1e-12),
        "python_tps": float(repeats) / max(python_seconds, 1e-12),
        "speedup": float(python_seconds / max(native_seconds, 1e-12)),
        "agreement": float(sum(int(a == b) for a, b in zip(native_preds, python_preds)) / max(1, repeats)),
        "cache_miss_proxy": float(diag["last_active_branches"]) / max(1.0, float(len(layer.branches))),
        "bytes": touched_bytes,
        "bandwidth_gbps": bytes_to_gb_per_s(touched_bytes, native_seconds),
        "native_ratio": float(diag["native_predict_ratio"]),
    }


def benchmark_cortical_topk(length: int, top_k: int, repeats: int) -> Dict[str, object]:
    memory = CorticalColumnSequenceMemory(vocab_size=256, cells_per_column=64, sparsity=0.25, seed=0)
    for token in range(48):
        memory.forward(token, learn=True)
    scores = np.linspace(0.0, 1.0, length, dtype=np.float32)
    native_preds, native_seconds = timed(lambda: [memory.topk_prediction_indices(scores, top_k=top_k, prefer_native=True) for _ in range(repeats)])
    python_preds, python_seconds = timed(lambda: [memory.topk_prediction_indices(scores, top_k=top_k, prefer_native=False) for _ in range(repeats)])
    agreement = float(sum(int(list(a) == list(b)) for a, b in zip(native_preds, python_preds)) / max(1, repeats))
    return {
        "mechanism": "cortical_topk",
        "native_tps": float(repeats) / max(native_seconds, 1e-12),
        "python_tps": float(repeats) / max(python_seconds, 1e-12),
        "speedup": float(python_seconds / max(native_seconds, 1e-12)),
        "agreement": agreement,
        "cache_miss_proxy": float(top_k) / max(1.0, float(length)),
        "bytes": float(length * 4),
        "bandwidth_gbps": bytes_to_gb_per_s(length * 4, native_seconds),
        "native_ratio": float(memory.metrics().get("native_topk_ratio", 0.0)),
    }


def benchmark_checkpoint_io(text: str, repeats: int = 3) -> Dict[str, object]:
    model = LSLCoreModel(vocab_size=512, runtime_profile="native_fast", seed=0)
    model.train_stream([text[:20000]], max_tokens=4096)
    save_seconds = []
    load_seconds = []
    compression_ratios = []
    file_bytes = 0
    with tempfile.TemporaryDirectory() as raw:
        path = os.path.join(raw, "checkpoint.lslb")
        for _ in range(repeats):
            _, save_elapsed = timed(lambda: model.save_binary(path))
            save_seconds.append(save_elapsed)
            file_bytes = os.path.getsize(path)
            compression_ratios.append(float(model.last_checkpoint_info.get("compression_ratio", 1.0)))
            _, load_elapsed = timed(lambda: LSLCoreModel.load(path))
            load_seconds.append(load_elapsed)
    avg_save = sum(save_seconds) / max(1, len(save_seconds))
    avg_load = sum(load_seconds) / max(1, len(load_seconds))
    save_gbps = bytes_to_gb_per_s(file_bytes, avg_save)
    load_gbps = bytes_to_gb_per_s(file_bytes, avg_load)
    return {
        "mechanism": "checkpoint_binary_io",
        "save_seconds": float(avg_save),
        "load_seconds": float(avg_load),
        "save_mb_s": save_gbps * 1000.0,
        "load_mb_s": load_gbps * 1000.0,
        "bandwidth_gbps": float((save_gbps + load_gbps) / 2.0),
        "file_bytes": float(file_bytes),
        "compression_ratio": float(sum(compression_ratios) / max(1, len(compression_ratios))),
        "runtime_profile": model.runtime_profile(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="vietnamese_small")
    parser.add_argument("--tokenizer-vocab", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--hippo-items", type=int, default=4096)
    parser.add_argument("--hippo-lookups", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--in-dim", type=int, default=2048)
    parser.add_argument("--active-count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--dendrite-branches", type=int, default=256)
    parser.add_argument("--dendrite-active-bits", type=int, default=16)
    parser.add_argument("--dendrite-outputs", type=int, default=8)
    parser.add_argument("--cortical-length", type=int, default=512)
    parser.add_argument("--cortical-topk", type=int, default=8)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    loader = DatasetLoader(ROOT)
    text = loader.load_text_splits(args.dataset, max_train_chars=40000, max_eval_chars=12000).train
    rows = [
        benchmark_tokenizer(text[:20000], int(args.tokenizer_vocab), int(args.max_tokens)),
        benchmark_hippocampus_lookup(int(args.hippo_items), int(args.hippo_lookups)),
        benchmark_sparse_batch(int(args.batch_size), int(args.out_dim), int(args.in_dim), int(args.active_count), int(args.repeats)),
        benchmark_dendrite(int(args.dendrite_branches), int(args.dendrite_active_bits), int(args.dendrite_outputs), int(args.repeats)),
        benchmark_cortical_topk(int(args.cortical_length), int(args.cortical_topk), int(args.repeats)),
        benchmark_checkpoint_io(text),
    ]
    payload = {
        "benchmark": "native_kernel_suite",
        "success": True,
        "config": vars(args),
        "rows": rows,
    }
    output = write_result(
        payload,
        benchmark="native_kernel_suite",
        dataset=args.dataset,
        seed=0,
        config=vars(args),
        output_path=args.json_output,
        results_root=args.results_root,
    )
    print(f"Result JSON: {output}")
    for row in rows:
        summary = []
        if "speedup" in row:
            summary.append(f"speedup={float(row['speedup']):.2f}x")
        if "agreement" in row:
            summary.append(f"agreement={float(row['agreement']):.3f}")
        if "native_ratio" in row:
            summary.append(f"native_ratio={float(row['native_ratio']):.3f}")
        if "cache_miss_proxy" in row:
            summary.append(f"cache_miss={float(row['cache_miss_proxy']):.6f}")
        summary.append(f"bandwidth={float(row['bandwidth_gbps']):.3f} GB/s")
        print(f"{row['mechanism']:<24} " + " ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
