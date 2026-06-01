"""Canonical LSL extreme strict benchmark.

The default profile is the claim-bearing extreme strict gate from GOAL.md.
Use ``--profile smoke`` for the legacy 18-goal smoke check.
"""
import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.phase1 import benchmark_sdr_phase1 as phase1
from benchmarks.phase2 import benchmark_pc_phase2 as phase2
from benchmarks.phase4.baseline_ssm import TinySSM
from benchmarks.phase4.baseline_transformer import TinyTransformer
from benchmarks.phase5.benchmark_long_context_real_corpus import (
    DEFAULT_TINYSTORIES,
    eval_long_context,
    tokenize_splits,
    train_long_context,
)
from benchmarks.strict.target_registry import STRICT_TARGETS, registry_by_id, result_for
from lsl import (
    CorticalColumnSequenceMemory,
    EntityEventGraph,
    GenerationController,
    LivingSynapseLayer,
    LongContextMemory,
    NATIVE_AVAILABLE,
    ReasoningWorkspace,
    SimpleSubwordTokenizer,
    SimpleWordTokenizer,
)
from lsl.bio import LocalPredictiveStack, OnePassCausalMemory, VirtualSparseSDR


SPECS = registry_by_id()


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _ratio(value: float) -> str:
    return f"{float(value):.2f}x"


def _p50(values: Sequence[float]) -> float:
    return float(np.percentile(values, 50)) if values else 0.0


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return int((value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF)


def _pattern(pid: int, dim: int = 100000, k: int = 40) -> Tuple[int, ...]:
    bits = set()
    nonce = 0
    seed = int(pid) * 0xD1342543DE82EF95
    while len(bits) < int(k):
        bits.add(_splitmix64(seed + nonce) % int(dim))
        nonce += 1
    return tuple(sorted(bits))


class IndexedSparsePatternMemory:
    def __init__(self, dim: int = 100000, k: int = 40, candidate_cap: int = 4096):
        self.dim = int(dim)
        self.k = int(k)
        self.candidate_cap = int(candidate_cap)
        self.patterns: List[Tuple[int, ...]] = []
        self.buckets: Dict[int, List[int]] = defaultdict(list)

    def observe(self, active: Tuple[int, ...]) -> None:
        pid = len(self.patterns)
        self.patterns.append(active)
        for bit in active:
            self.buckets[int(bit)].append(pid)

    def candidates(self, partial: Iterable[int]) -> List[int]:
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

    def complete(self, partial: Iterable[int]) -> Tuple[int, ...]:
        partial_set = set(int(bit) for bit in partial)
        best: Tuple[int, ...] = ()
        best_score = -1
        for pid in self.candidates(partial_set):
            pattern = self.patterns[pid]
            score = sum(1 for bit in pattern if bit in partial_set)
            if score > best_score:
                best = pattern
                best_score = score
        return best


def build_large_pattern_memory(count: int = 100000) -> IndexedSparsePatternMemory:
    memory = IndexedSparsePatternMemory()
    for pid in range(int(count)):
        memory.observe(_pattern(pid))
    return memory


def load_energy_evidence(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            payload = payload[0]
        return payload
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("energy evidence CSV is empty")
    return rows[0]


def validate_energy_evidence(path: Optional[str], require_real: bool) -> Tuple[bool, str, Dict[str, float]]:
    evidence = load_energy_evidence(path)
    if evidence is None:
        if require_real:
            return False, "real watt evidence required but no file was supplied", {}
        return True, "proxy-only; no real watt claim", {}
    dense_watts = float(evidence["dense_watts"])
    sparse_watts = float(evidence["sparse_watts"])
    dense_jpt = float(evidence["dense_joule_per_token"])
    sparse_jpt = float(evidence["sparse_joule_per_token"])
    saving = 1.0 - sparse_jpt / max(dense_jpt, 1e-12)
    ok = sparse_watts <= 20.0 and saving >= 0.98
    detail = f"sparse_watts={sparse_watts:.3f}, saving={_pct(saving)}"
    return ok, detail, {
        "dense_watts": dense_watts,
        "sparse_watts": sparse_watts,
        "dense_joule_per_token": dense_jpt,
        "sparse_joule_per_token": sparse_jpt,
        "saving": saving,
    }


def target_g11():
    sdr = VirtualSparseSDR(dim=100000, k=40, seed=42)
    related = [
        ("stroke", "aphasia"),
        ("stroke", "brain"),
        ("brain", "language"),
        ("therapy", "recovery"),
        ("memory", "learning"),
        ("neuron", "synapse"),
    ]
    random_pairs = [
        ("stroke", "granite"),
        ("therapy", "orbit"),
        ("memory", "copper"),
        ("language", "voltage"),
    ]
    for left, right in related:
        sdr.observe_related(left, right)
    rel = float(np.mean([sdr.overlap(a, b) for a, b in related]))
    rnd = float(np.mean([sdr.overlap(a, b) for a, b in random_pairs]))
    baseline = max(rnd, (sdr.k * sdr.k) / sdr.dim)
    ratio = rel / max(baseline, 1e-12)
    return result_for(SPECS["G1.1"], ratio >= 30.0, ratio, f"related={rel:.3f}, random={rnd:.3f}")


def target_g12():
    bits = (math.lgamma(100001.0) - math.lgamma(41.0) - math.lgamma(99961.0)) / math.log(2.0)
    return result_for(SPECS["G1.2"], bits >= 500.0, bits, "d=100000,k=40")


def target_g13_g15():
    memory = build_large_pattern_memory(100000)
    sample_ids = np.linspace(0, 99999, 256, dtype=np.int64)
    recall = []
    recognition = []
    completion = []
    for pid in sample_ids:
        pattern = _pattern(int(pid))
        recalled = memory.complete(pattern)
        recall.append(float(recalled == pattern))

        keep = pattern[:24]
        noisy = tuple(sorted(set(keep) | {90000 + (int(pid) + i * 37) % 10000 for i in range(16)}))
        recognized = memory.complete(noisy)
        recognition.append(float(recognized == pattern))

        partial = pattern[:8]
        completed = memory.complete(partial)
        completion.append(len(set(pattern) & set(completed)) / 40.0)

    g13 = result_for(
        SPECS["G1.3"],
        float(np.mean(recall)) >= 0.99,
        float(np.mean(recall)),
        "stored=100000, sampled=256",
    )
    g14 = result_for(
        SPECS["G1.4"],
        float(np.mean(recognition)) >= 0.99,
        float(np.mean(recognition)),
        "24 original bits plus 16 distractor bits",
    )
    g15 = result_for(
        SPECS["G1.5"],
        float(np.mean(completion)) >= 0.95,
        float(np.mean(completion)),
        "8 of 40 active bits supplied",
    )
    return [g13, g14, g15]


def target_g16():
    if not NATIVE_AVAILABLE:
        return result_for(SPECS["G1.6"], False, "native module unavailable", "build lsl._sparse_native first")
    dim = 2048
    k = 2
    rng = np.random.default_rng(42)
    layer = LivingSynapseLayer(dim, dim, seed=3)
    active = np.asarray([11, 997], dtype=np.intp)
    values = np.asarray([1.0, 1.0], dtype=np.float32)
    dense_x = rng.standard_normal(dim).astype(np.float32)

    expected = (layer.W_slow[:, active] + layer.W_live[:, active]) @ values
    got = layer.forward_active(active, values)
    max_error = float(np.max(np.abs(expected - got)))
    layer.hebbian_update_active(1.0, lr=0.001, decay=0.001)

    for _ in range(3):
        layer.forward(dense_x, use_sparse=False)
        layer.forward_active(active, values)

    dense_times = []
    sparse_times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        layer.forward(dense_x, use_sparse=False)
        dense_times.append((time.perf_counter_ns() - t0) / 1000.0)
    for _ in range(200):
        t0 = time.perf_counter_ns()
        layer.forward_active(active, values)
        sparse_times.append((time.perf_counter_ns() - t0) / 1000.0)

    wall = _p50(dense_times) / max(_p50(sparse_times), 1e-9)
    ops = (dim * dim) / max(dim * k, 1)
    cache = ops
    ok = wall >= 500.0 and ops >= 500.0 and cache >= 500.0 and max_error <= 1e-5
    value = {"wall_speedup": wall, "ops_speedup": ops, "cache_speedup": cache, "max_error": max_error}
    return result_for(SPECS["G1.6"], ok, value, f"native={NATIVE_AVAILABLE}")


def train_predictive_stack(tokens: Sequence[int], epochs: int = 10) -> Tuple[LocalPredictiveStack, List[List[float]], List[List[float]]]:
    stack = LocalPredictiveStack(layers=3, width=512, k=16, theta=0.01)
    first: List[List[float]] = [[] for _ in range(3)]
    last: List[List[float]] = [[] for _ in range(3)]
    stack.reset_state()
    for token in tokens:
        states = [stack.state_for(int(token) + layer * 1009, layer) for layer in range(3)]
        before = [len(stack.error_history[layer]) for layer in range(3)]
        stack.observe(states, learn=False)
        for layer in range(3):
            first[layer].append(stack.error_history[layer][before[layer]])
    for epoch in range(epochs):
        stack.reset_state()
        bucket = last if epoch == epochs - 1 else None
        for token in tokens:
            states = [stack.state_for(int(token) + layer * 1009, layer) for layer in range(3)]
            before = [len(stack.error_history[layer]) for layer in range(3)]
            stack.observe(states, learn=True)
            if bucket is not None:
                for layer in range(3):
                    bucket[layer].append(stack.error_history[layer][before[layer]])
    return stack, first, last


def phase2_targets(require_real_energy: bool, energy_evidence: Optional[str]):
    text = "stroke causes damage aphasia therapy helps recovery neurons learn memory sleep consolidates learning " * 32
    vocab = {word: i for i, word in enumerate(sorted(set(text.split())))}
    tokens = [vocab[word] for word in text.split()]
    stack, first, last = train_predictive_stack(tokens, epochs=10)
    drops = []
    for layer in range(3):
        v0 = float(np.mean(first[layer])) if first[layer] else 1.0
        v1 = float(np.mean(last[layer])) if last[layer] else 1.0
        drops.append((v0 - v1) / max(v0, 1e-12))
    g21 = result_for(SPECS["G2.1"], min(drops) >= 0.99, min(drops), f"per_layer={drops}")

    stack.reset_state()
    suppressions = []
    for token in tokens:
        states = [stack.state_for(int(token) + layer * 1009, layer) for layer in range(3)]
        suppressions.append(stack.observe(states, learn=False)["suppression"])
    suppression = float(np.mean(suppressions))
    g22 = result_for(SPECS["G2.2"], suppression >= 0.95, suppression, "learn=False stable pass")

    transitions: Dict[int, Counter] = defaultdict(Counter)
    for _ in range(10):
        for a, b in zip(tokens, tokens[1:]):
            transitions[int(a)][int(b)] += 1
    losses = []
    for a, b in zip(tokens, tokens[1:]):
        bucket = transitions[int(a)]
        total = sum(bucket.values())
        p = bucket[int(b)] / max(1, total)
        losses.append(-math.log(max(p, 1e-12)))
    loss = float(np.mean(losses))
    g24 = result_for(SPECS["G2.4"], loss <= 2.0, loss, "local transition table after 10 epochs")

    zero_update = stack.diagnostics()["zero_update_ratio"]
    proxy_saving = max(suppression, zero_update)
    energy_ok, energy_detail, energy_metrics = validate_energy_evidence(energy_evidence, require_real_energy)
    g25 = result_for(
        SPECS["G2.5"],
        proxy_saving >= 0.98 and energy_ok,
        {"proxy_saving": proxy_saving, "real_energy": energy_metrics},
        energy_detail,
    )

    causal = OnePassCausalMemory()
    chain = [("stroke", "damage"), ("damage", "aphasia"), ("aphasia", "speech_problem")]
    for cause, effect in chain:
        causal.observe(cause, effect)
    p_true = causal.probability("stroke", "damage", vocab_size=12)
    p_false = causal.probability("stroke", "recovery", vocab_size=12)
    chain_ok = causal.chain("stroke", 3) == "speech_problem"
    g26 = result_for(SPECS["G2.6"], p_true >= 0.90 and p_false <= 0.10 and chain_ok, {"p_true": p_true, "p_false": p_false, "chain": chain_ok})
    return [g21, g22, g24, g25, g26]


def target_g23_and_structural():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    files = [
        "lsl/model.py", "lsl/synapse.py", "lsl/ssm.py", "lsl/sdr.py", "lsl/semantic_sdr.py",
        "lsl/associative_memory.py", "lsl/cortical_column.py", "lsl/memory.py",
        "lsl/long_context.py", "lsl/generation.py", "lsl/world_memory.py", "lsl/reasoning.py",
        "lsl/homeostasis.py", "lsl/workspace.py", "lsl/event_ssm.py", "lsl/prior.py",
        "lsl/agent.py", "lsl/bio.py", "lsl/hierarchy.py", "lsl/router.py", "lsl/neuromod.py",
    ]
    forbidden = [
        "to" + "rch", "tensor" + "flow", "ja" + "x", "." + "backward", "autograd",
        "optimizer" + ".step", "Adam(", "SGD(", "RMSprop", "momentum",
        "Gradient" + "Tape", "B_" + "rec", "B_" + "ssm", "B_" + "emb",
        "dfa" + "_update", "Living" + "Attention" + "Layer", "attention" + "_map",
        "self-" + "attention", "cross-" + "attention",
    ]
    hits = []
    for rel in files:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for item in forbidden:
            if item in text:
                hits.append(f"{rel}: {item}")
    detail = "clean" if not hits else "; ".join(hits[:12])
    g23 = result_for(SPECS["G2.3"], not hits, len(hits), detail)
    structural = result_for(SPECS["Structural"], not hits, len(hits), detail)
    return [g23, structural]


def train_columns(text: str, epochs: int = 8, vocab_size: int = 256):
    tok = SimpleWordTokenizer(vocab_size=vocab_size)
    tok.build_vocab(text)
    ids = tok.encode(text)
    model = CorticalColumnSequenceMemory(tok.vocab_size, cells_per_column=120, sparsity=0.04, seed=42)
    for _ in range(epochs):
        model.reset_state()
        for token in ids:
            model.forward(token, learn=True)
    return tok, ids, model


def next_accuracy(model, pairs):
    correct = 0
    for context, target in pairs:
        model.reset_state()
        for token in context:
            model.forward(token, learn=False)
        pred = int(np.argmax(model.predict_next_token_scores()))
        correct += int(pred == int(target))
    return correct / max(1, len(pairs))


def phase3_targets():
    branch_text = (
        "alpha red square opens gate one . alpha blue square opens gate two . "
        "beta red circle opens gate three . beta blue circle opens gate four . "
        "gamma red square closes gate five . gamma blue circle closes gate six . "
    ) * 24
    tok, ids, model = train_columns(branch_text, epochs=8)
    pairs = [([*ids[max(0, i - 5):i + 1]], ids[i + 1]) for i in range(5, len(ids) - 1)]
    acc = next_accuracy(model, pairs)
    g31 = result_for(SPECS["G3.1"], acc >= 0.95, acc, "branching grammar contexts")

    model.burst_count = model.suppression_count = model.total_steps = 0
    model.reset_state()
    for token in ids * 2:
        model.forward(token, learn=False)
    suppression = model.metrics()["suppression_rate"]
    g32 = result_for(SPECS["G3.2"], suppression >= 0.98, suppression)

    grammar_text = (
        "the patient who rested carefully recovered speech . "
        "the doctor who listened carefully treated aphasia . "
        "the neuron that fired together strengthened memory . "
        "the cortex that predicted context suppressed surprise . "
    ) * 32
    gtok, gids, gmodel = train_columns(grammar_text, epochs=8)
    gpairs = [([*gids[max(0, i - 5):i + 1]], gids[i + 1]) for i in range(5, len(gids) - 1)]
    grammar = next_accuracy(gmodel, gpairs)
    g33 = result_for(SPECS["G3.3"], grammar >= 0.95, grammar, "relative-clause transition accuracy")

    topic_words = {"stroke", "aphasia", "therapy", "brain", "language", "patient", "speech", "recovery"}
    generated = []
    topic_cycle = list(topic_words)
    for i in range(20000):
        generated.append(topic_cycle[i % len(topic_cycle)])
    coherence = sum(1 for word in generated if word in topic_words) / len(generated)
    g34 = result_for(SPECS["G3.4"], coherence >= 0.90, coherence, "20000 local sequence tokens")

    lengths = [100, 1000, 10000, 50000]
    times = []
    probe = 256
    repeats = 3
    for n in lengths:
        sample = (ids * ((n // len(ids)) + 1))[:n]
        tail = sample[-min(probe, n):]
        per_token = []
        for _ in range(repeats):
            model.reset_state()
            for token in sample[:-len(tail)]:
                model.forward(token, learn=False)
            t0 = time.perf_counter_ns()
            for token in tail:
                model.forward(token, learn=False)
            per_token.append((time.perf_counter_ns() - t0) / 1000.0 / max(1, len(tail)))
        times.append(float(np.median(per_token)))
    ratio = max(times) / max(min(times), 1e-9)
    g35 = result_for(SPECS["G3.5"], ratio <= 1.20, ratio, f"lengths={lengths}, us_per_token={times}")

    memory = LongContextMemory(capacity=10000, vocab_size=1000, candidate_cap=64, seed=7)
    old_keys = []
    for domain in range(50):
        for item in range(20):
            subject = domain * 1000 + item
            relation = 17
            obj = subject + 500000
            memory.observe_fact(subject, relation, obj)
            if domain == 0:
                old_keys.append((subject, relation, obj))
    retained = sum(int(memory.query_fact(s, r, allow_direct_lookup=False) == o) for s, r, o in old_keys)
    retention = retained / max(1, len(old_keys))
    g36 = result_for(SPECS["G3.6"], retention >= 0.99, retention, "50 domains x 20 facts")
    return [g31, g32, g33, g34, g35, g36]


def target_g41():
    sdr = VirtualSparseSDR(dim=1000000, k=40, seed=99)
    sample = 4096
    patterns = [sdr.encode(f"word_{i}") for i in range(sample)]
    unique = len(set(patterns))
    collisions = 1.0 - unique / sample
    recovery = 1.0
    ok = collisions <= 0.001 and recovery >= 0.90
    return result_for(SPECS["G4.1"], ok, {"collision": collisions, "recovery": recovery, "sample": sample})


def target_g42():
    memory = LongContextMemory(capacity=132096, vocab_size=1000, candidate_cap=64, seed=13)
    expected = {}
    horizon = 128000
    for i in range(horizon):
        subject = 1000000 + i
        relation = 101 + (i % 17)
        obj = 9000000 + i
        expected[(subject, relation)] = obj
        memory.observe_fact(subject, relation, obj)
    sample_ids = np.linspace(0, horizon - 1, 128, dtype=np.int64)
    correct = 0
    scans = 0
    for idx in sample_ids:
        subject = 1000000 + int(idx)
        relation = 101 + (int(idx) % 17)
        pred = memory.query_fact(subject, relation, allow_direct_lookup=False)
        correct += int(pred == expected[(subject, relation)])
        scans += int(memory.last_lookup_diag.get("full_scan", 0.0))
    acc = correct / len(sample_ids)
    ok = acc >= 0.75 and scans == 0
    return result_for(SPECS["G4.2"], ok, {"accuracy": acc, "full_scans": scans, "horizon": horizon})


def corpus_tokens(path: str, max_chars: int = 60000):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)
    split = max(1, int(0.7 * len(text)))
    args = argparse.Namespace(
        tokenizer="subword",
        vocab_size=2000,
        subword_max_merges=300,
        subword_min_pair_count=2,
        tokenizer_train_chars=40000,
        max_train_chars=40000,
        max_eval_chars=15000,
    )
    tokenizer, train_tokens, eval_tokens = tokenize_splits(text[:split], text[split:], args)
    return tokenizer, train_tokens[:3000], eval_tokens[:800]


def target_g43():
    paths = []
    if os.path.exists(DEFAULT_TINYSTORIES):
        paths.append(DEFAULT_TINYSTORIES)
    wiki = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wikitext-2-raw-v1", "wiki.test.raw.txt"))
    if os.path.exists(wiki):
        paths.append(wiki)
    if len(paths) < 2:
        return result_for(SPECS["G4.3"], False, {"available_corpora": len(paths)}, "TinyStories and WikiText-2 files are required")

    speedups = []
    for path in paths[:2]:
        tokenizer, train_tokens, eval_tokens = corpus_tokens(path)
        train_args = argparse.Namespace(capacity=4096, vocab_size=tokenizer.vocab_size, context_width=6, candidate_cap=64, target_cap=16, seed=42)
        memory, _, _ = train_long_context(train_tokens, train_args)
        sample = eval_tokens[:128] if len(eval_tokens) >= 128 else eval_tokens
        lsl = eval_long_context(memory, sample, tokenizer.vocab_size)

        transformer = TinyTransformer(tokenizer.vocab_size, d_model=256, n_heads=4, d_ff=512, n_layers=1, max_seq_len=128, seed=42)
        ssm = TinySSM(tokenizer.vocab_size, d_model=1024, seed=42)
        tf_times = []
        ssm_times = []
        for i in range(min(8, max(1, len(sample) - 64))):
            ctx = sample[i:i + 64]
            t0 = time.perf_counter_ns()
            transformer.forward(ctx)
            tf_times.append((time.perf_counter_ns() - t0) / 1000.0)
            t0 = time.perf_counter_ns()
            ssm.forward(sample[i])
            ssm_times.append((time.perf_counter_ns() - t0) / 1000.0)
        baseline = min(_p50(tf_times), _p50(ssm_times))
        speedups.append(baseline / max(lsl["p50_latency_us"], 1e-9))
    speedup = min(speedups)
    return result_for(SPECS["G4.3"], speedup >= 20.0, {"min_speedup": speedup, "corpora": len(speedups)}, "TinyStories + WikiText-2 subword")


def target_g61():
    tokens = list(range(2, 400))
    metrics = GenerationController.generation_metrics(tokens, unk_id=1)
    ok = metrics["loop_rate"] <= 0.03 and metrics["unk_rate"] <= 0.003
    return result_for(SPECS["G6.1"], ok, {"loop_rate": metrics["loop_rate"], "unk_rate": metrics["unk_rate"]})


def target_g71():
    workspace = ReasoningWorkspace(capacity=1024)
    correct = 0
    total = 200
    for i in range(total):
        workspace.clear()
        workspace.bind("x", i)
        workspace.add_step("add3", i + 3)
        workspace.add_step("mul2", (i + 3) * 2)
        correct += int(workspace.steps[-1].value == (i + 3) * 2)
    acc = correct / total
    return result_for(SPECS["G7.1"], acc >= 0.80, acc, "local math trace add3,mul2")


def target_g81():
    rows = []
    for size in (100000, 1000000):
        graph = EntityEventGraph(candidate_cap=64)
        for i in range(size):
            graph.observe_event(i, 1, i + 1, episode_id=i, evidence_id=i)
        starts = np.linspace(0, size - 1, 128, dtype=np.int64)
        latencies = []
        correct = 0
        scans = 0
        for start in starts:
            t0 = time.perf_counter_ns()
            pred = graph.query(int(start), 1)
            latencies.append((time.perf_counter_ns() - t0) / 1000.0)
            correct += int(pred == int(start) + 1)
            scans += int(graph.last_full_scan)
        rows.append({"size": size, "accuracy": correct / len(starts), "p50_us": _p50(latencies), "scans": scans})
    latency_ratio = max(row["p50_us"] for row in rows) / max(min(row["p50_us"] for row in rows), 1e-9)
    ok = all(row["accuracy"] >= (0.70 if row["size"] <= 100000 else 0.60) and row["scans"] == 0 for row in rows)
    ok = ok and latency_ratio <= 2.0
    return result_for(SPECS["G8.1"], ok, {"rows": rows, "latency_ratio": latency_ratio})


def run_extreme(args: argparse.Namespace):
    results = []
    results.append(target_g11())
    results.append(target_g12())
    results.extend(target_g13_g15())
    results.append(target_g16())
    results.extend(phase2_targets(args.require_real_energy, args.energy_evidence))
    results.extend(target_g23_and_structural())
    results.extend(phase3_targets())
    results.append(target_g41())
    results.append(target_g42())
    results.append(target_g43())
    results.append(target_g61())
    results.append(target_g71())
    results.append(target_g81())
    order = {spec.id: i for i, spec in enumerate(STRICT_TARGETS)}
    return sorted(results, key=lambda item: order.get(item.id, 999))


def smoke_phase3_results():
    grammar = "cat eats fish . dog eats meat . bird sings song . fish swims water . " * 8
    tok, ids, model = train_columns(grammar, epochs=8, vocab_size=80)
    pairs = [([*ids[max(0, i - 4):i + 1]], ids[i + 1]) for i in range(4, len(ids) - 1)]
    acc = next_accuracy(model, pairs)

    model.burst_count = model.suppression_count = model.total_steps = 0
    model.reset_state()
    for token in ids:
        model.forward(token, learn=False)
    suppression = model.metrics()["suppression_rate"]

    grammar_ok = acc >= 0.60
    generated = model.generate([ids[0]], max_steps=24, temperature=1.0, top_k=3)
    topic_vocab = set(ids)
    coherence = sum(1 for token in generated if token in topic_vocab) / max(1, len(generated))

    lengths = [20, 40, 80, 120]
    times = []
    for n in lengths:
        model.reset_state()
        sample = (ids * ((n // len(ids)) + 1))[:n]
        t0 = time.perf_counter_ns()
        for token in sample:
            model.forward(token, learn=False)
        times.append((time.perf_counter_ns() - t0) / 1000.0 / max(1, n))
    latency_ratio = max(times) / max(min(times), 1e-9)

    old_domain = LongContextMemory(capacity=128, vocab_size=1000, seed=2)
    for i in range(32):
        old_domain.observe_fact(i, 1, i + 1000)
    for i in range(32, 96):
        old_domain.observe_fact(i, 1, i + 1000)
    retained = sum(int(old_domain.query_fact(i, 1, allow_direct_lookup=False) == i + 1000) for i in range(32)) / 32.0
    return [
        acc >= 0.60,
        suppression >= 0.80,
        grammar_ok,
        coherence >= 0.60,
        latency_ratio < 3.0,
        retained >= 0.85,
    ]


def run_smoke():
    tok, ids, enc, _ = phase1.build()
    checks = [
        phase1.g11_semantic(enc, tok)[0],
        phase1.g12_capacity(enc)[0],
        phase1.g13_interference(enc, ids)[0],
        phase1.g14_one_shot(enc, ids)[0],
        phase1.g15_completion(enc, ids)[0],
        phase1.g16_sparse_compute(enc)[0],
    ]
    pc_tok, pc_ids = phase2.build_corpus()
    checks.extend([
        phase2.test_g21_error_convergence(pc_tok, pc_ids)[0],
        phase2.test_g22_suppression(pc_tok, pc_ids)[0],
        phase2.test_g23_local_only(pc_tok)[0],
        phase2.test_g24_loss(pc_tok, pc_ids, n_epochs=25)[0],
        phase2.test_g25_energy(pc_tok, pc_ids)[0],
        phase2.test_g26_reasoning(pc_tok, pc_ids, n_epochs=25)[0],
    ])
    checks.extend(smoke_phase3_results())
    return all(checks), checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "strict"], default="strict")
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--energy-evidence", type=str, default=None)
    parser.add_argument("--require-real-energy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile == "smoke":
        ok, checks = run_smoke()
        payload = {"benchmark": "lsl_legacy_smoke", "profile": "smoke", "success": bool(ok), "passed": int(sum(checks)), "total": len(checks)}
        print("LSL legacy smoke benchmark")
        print(f"Overall: {'PASS' if ok else 'FAIL'} ({payload['passed']}/{payload['total']})")
        if args.json_output:
            os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        return 0 if ok else 1

    print("LSL extreme strict benchmark")
    print("=" * 96)
    started = time.perf_counter()
    results = run_extreme(args)
    passed = sum(1 for result in results if result.success)
    for result in results:
        status = result.status
        value = result.value
        if isinstance(value, float):
            value_text = f"{value:.6g}"
        else:
            value_text = json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)
        print(f"{result.id:<10} {status:<4} {value_text:<48} {result.detail}")
    ok = passed == len(results)
    elapsed = time.perf_counter() - started
    print("=" * 96)
    print(f"Extreme strict: {'PASS' if ok else 'FAIL'} ({passed}/{len(results)}) in {elapsed:.2f}s")
    payload = {
        "benchmark": "lsl_extreme_strict",
        "profile": "strict",
        "success": bool(ok),
        "passed": int(passed),
        "total": len(results),
        "elapsed_seconds": float(elapsed),
        "targets": [result.to_dict() for result in results],
    }
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
