"""Strict Phase 1 SDR benchmark.

Targets from GOAL.md:
  G1.1 semantic overlap related/random >= 3x
  G1.2 capacity log2 C(1024,20) >= 130 bits
  G1.3 retention >= 90% and interference <= 10% after 80 patterns
  G1.4 one-shot recognition >= 80%
  G1.5 completion >= 70% from 50% active-bit mask
  G1.6 sparse compute >= 40x by ops and wall-clock full forward
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import SimpleWordTokenizer, SemanticSDREncoder, hamming_overlap
from lsl import capacity_stats
from lsl.associative_memory import SparseAssociativeMemory
from lsl.synapse import LivingSynapseLayer


CORPUS = """
the patient had a stroke and developed aphasia .
stroke damages the brain and causes language problems .
aphasia affects speech and language after brain damage .
the doctor examined the patient with aphasia .
language therapy helps patients recover from aphasia .
the brain controls speech and language processing .
stroke is a sudden interruption of blood flow to the brain .
patients with stroke often have difficulty speaking .
the patient recovered language skills after therapy .
brain damage from stroke can cause aphasia .
speech therapy is effective for aphasia treatment .
the neuron is the basic unit of the brain .
neurons communicate through synapses in the brain .
learning changes synaptic connections between neurons .
memory is stored in patterns of neural activity .
the cortex processes language and cognitive functions .
reading requires visual processing and language comprehension .
writing involves motor control and language production .
stroke and brain damage affect language and speech .
aphasia is caused by stroke and brain injury .
brain plasticity helps language recovery after stroke .
therapy improves language and speech in aphasia patients .
the doctor studies brain regions for language and memory .
stroke patients with aphasia need speech therapy .
brain imaging shows stroke damage in language regions .
neurons in language regions are affected by stroke .
memory and learning depend on synaptic connections in the brain .
aphasia therapy focuses on language and communication skills .
the brain recovers language skills through neural plasticity .
patients with brain damage show language and memory problems .
stroke causes damage to neurons and synapses in the brain .
language and speech are processed by cortical neurons .
therapy sessions strengthen synaptic connections for language recovery .
brain plasticity allows neurons to learn new language patterns .
stroke disrupts blood flow and damages brain neurons .
aphasia affects language production and comprehension after stroke .
memory for language patterns depends on cortical neural activity .
learning language requires repeated activation of brain neurons .
"""


def build():
    tok = SimpleWordTokenizer(vocab_size=220)
    tok.build_vocab(CORPUS)
    ids = tok.encode(CORPUS)
    enc = SemanticSDREncoder(
        vocab_size=tok.vocab_size,
        sdr_dim=1024,
        sparsity=0.02,
        embed_dim=32,
        seed=42,
    )
    enc.fit([ids], window=6, verbose=False)
    loaded = enc.load_builtin_embeddings(tok.word_to_id)
    return tok, ids, enc, loaded


def g11_semantic(enc, tok):
    related = [
        ("stroke", "brain"), ("stroke", "aphasia"), ("brain", "damage"),
        ("language", "speech"), ("therapy", "patient"), ("memory", "learning"),
        ("neuron", "synapse"), ("brain", "learning"),
    ]
    random_pairs = [
        ("stroke", "table"), ("therapy", "water"), ("memory", "pressure"),
        ("language", "blood"), ("doctor", "river"), ("aphasia", "ball"),
    ]
    vocab = tok.word_to_id
    rel = [enc.semantic_overlap(vocab[a], vocab[b]) for a, b in related
           if a in vocab and b in vocab]
    rnd = [enc.semantic_overlap(vocab[a], vocab[b]) for a, b in random_pairs
           if a in vocab and b in vocab]
    mean_rel = float(np.mean(rel))
    mean_rnd = float(np.mean(rnd)) if rnd else enc.random_baseline_overlap()
    ratio = mean_rel / max(mean_rnd, 0.1)
    ok = ratio >= 3.0
    print(f"G1.1 semantic overlap: related={mean_rel:.2f}, random={mean_rnd:.2f}, ratio={ratio:.2f}x -> {'PASS' if ok else 'FAIL'}")
    return ok, ratio


def g12_capacity(enc):
    stats = capacity_stats(enc.sdr_dim, enc.k)
    bits = stats["log2_capacity"]
    ok = bits >= 130.0
    print(f"G1.2 capacity: log2 C({enc.sdr_dim},{enc.k})={bits:.2f} -> {'PASS' if ok else 'FAIL'}")
    return ok, bits


def g13_interference(enc, ids):
    unique = list(dict.fromkeys(ids))[:80]
    mem = SparseAssociativeMemory(enc.sdr_dim, enc.k)
    patterns = [enc.encode(t) for t in unique]
    original = patterns[0].copy()
    for pattern in patterns:
        mem.observe(pattern)
    retrieved = mem.complete(original, k=enc.k)
    retention = hamming_overlap(original, retrieved) / enc.k
    interference = 1.0 - retention
    ok = retention >= 0.90 and interference <= 0.10
    print(f"G1.3 interference: retention={100*retention:.1f}%, interference={100*interference:.1f}% -> {'PASS' if ok else 'FAIL'}")
    return ok, retention


def g14_one_shot(enc, ids):
    unique = list(dict.fromkeys(ids))[:80]
    stored = {tid: enc.encode(tid) for tid in unique}
    rng = np.random.default_rng(7)
    correct = 0
    for tid, pattern in stored.items():
        noisy = pattern.copy()
        active = np.where(noisy > 0.5)[0]
        noisy[rng.choice(active, size=max(1, enc.k // 5), replace=False)] = 0.0
        scores = {cand: hamming_overlap(noisy, cand_pattern)
                  for cand, cand_pattern in stored.items()}
        if max(scores, key=scores.get) == tid:
            correct += 1
    acc = correct / len(stored)
    ok = acc >= 0.80
    print(f"G1.4 one-shot recognition: accuracy={100*acc:.1f}% -> {'PASS' if ok else 'FAIL'}")
    return ok, acc


def g15_completion(enc, ids):
    unique = list(dict.fromkeys(ids))[:60]
    mem = SparseAssociativeMemory(enc.sdr_dim, enc.k)
    patterns = []
    for tid in unique:
        pattern = enc.encode(tid)
        patterns.append(pattern)
        mem.observe(pattern)
    acc = mem.completion_accuracy(np.asarray(patterns), mask_fraction=0.5, seed=11)
    ok = acc >= 0.70
    print(f"G1.5 pattern completion: accuracy={100*acc:.1f}% -> {'PASS' if ok else 'FAIL'}")
    return ok, acc


def g16_sparse_compute(enc):
    d, k = enc.sdr_dim, enc.k
    rng = np.random.default_rng(0)
    dense_layer = LivingSynapseLayer(d, d, seed=1)
    sparse_layer = LivingSynapseLayer(d, d, seed=1)
    x_dense = rng.standard_normal(d).astype(np.float32)
    x_sparse = np.zeros(d, dtype=np.float32)
    x_sparse[rng.choice(d, k, replace=False)] = 1.0

    for _ in range(5):
        dense_layer.forward(x_dense)
        sparse_layer.forward(x_sparse, use_sparse=True)

    t0 = time.perf_counter()
    for _ in range(100):
        dense_layer.forward(x_dense)
    dense_us = (time.perf_counter() - t0) / 100.0 * 1e6

    t0 = time.perf_counter()
    for _ in range(1000):
        sparse_layer.forward(x_sparse, use_sparse=True)
    sparse_us = (time.perf_counter() - t0) / 1000.0 * 1e6

    dense_ops = dense_layer.last_forward_ops["ops"]
    sparse_ops = sparse_layer.last_forward_ops["ops"]
    ops_speedup = dense_ops / sparse_ops
    wall_speedup = dense_us / max(sparse_us, 1e-9)
    ok = ops_speedup >= 40.0 and wall_speedup >= 40.0
    print(f"G1.6 sparse compute: ops={ops_speedup:.1f}x, wall={wall_speedup:.1f}x -> {'PASS' if ok else 'FAIL'}")
    return ok, min(ops_speedup, wall_speedup)


def main():
    np.random.seed(42)
    tok, ids, enc, loaded = build()
    print("Strict Phase 1 SDR Benchmark")
    print(f"vocab={tok.vocab_size}, tokens={len(ids)}, builtin_embeddings={loaded}, d={enc.sdr_dim}, k={enc.k}")
    results = {
        "G1.1": g11_semantic(enc, tok),
        "G1.2": g12_capacity(enc),
        "G1.3": g13_interference(enc, ids),
        "G1.4": g14_one_shot(enc, ids),
        "G1.5": g15_completion(enc, ids),
        "G1.6": g16_sparse_compute(enc),
    }
    passed = sum(1 for ok, _ in results.values() if ok)
    print(f"Phase 1 result: {passed}/6")
    return 0 if passed == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
