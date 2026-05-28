"""Canonical strict GOAL.md benchmark for all 18 goals."""
import os
import time

import numpy as np

from benchmarks.phase1 import benchmark_sdr_phase1 as phase1
from benchmarks.phase2 import benchmark_pc_phase2 as phase2
from lsl import CorticalColumnSequenceMemory, SimpleWordTokenizer
from lsl import SemanticSDREncoder


GRAMMAR_CORPUS = (
    "the cat eats fish . the dog eats meat . the bird flies in the sky . "
    "the fish swims in the water . the cat sleeps on the bed . "
    "the dog runs in the park . the bird sings a song . the fish jumps out of water . "
    "the cat eats fish . the dog eats meat . the bird flies in the sky . "
    "the fish swims in the water . the cat sleeps on the bed . "
    "the dog runs in the park . the bird sings a song . the fish jumps out of water . "
)

MEDICAL_GEN_CORPUS = (
    "stroke causes aphasia . brain damage affects language . "
    "therapy helps patient recover language . stroke damages brain neurons . "
    "aphasia affects speech language . therapy helps aphasia patient . "
) * 6


def train_columns(text, vocab_size=80, epochs=10, cells=100, sparsity=0.02):
    tok = SimpleWordTokenizer(vocab_size=vocab_size)
    tok.build_vocab(text)
    ids = tok.encode(text)
    model = CorticalColumnSequenceMemory(
        vocab_size=tok.vocab_size,
        cells_per_column=cells,
        sparsity=sparsity,
        seed=42,
    )
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
        correct += int(pred == target)
    return correct / max(1, len(pairs))


def phase3_g31():
    tok, ids, model = train_columns(GRAMMAR_CORPUS, epochs=8)
    pairs = [([*ids[max(0, i - 4):i + 1]], ids[i + 1])
             for i in range(4, len(ids) - 1)]
    acc = next_accuracy(model, pairs)
    ok = acc >= 0.60
    return ok, f"{100*acc:.1f}%"


def phase3_g32():
    tok, ids, model = train_columns(GRAMMAR_CORPUS * 2, epochs=10)
    model.burst_count = model.suppression_count = model.total_steps = 0
    model.reset_state()
    for token in ids:
        model.forward(token, learn=False)
    silent = model.metrics()["suppression_rate"]
    ok = silent >= 0.80
    return ok, f"{100*silent:.1f}%"


def phase3_g33():
    corpus = (
        "cat eats fish . dog eats meat . bird sings song . fish swims water . "
        "cat drinks water . dog chases ball . bird builds nest . fish jumps water . "
    ) * 8
    tok, ids, model = train_columns(corpus, vocab_size=40, epochs=8, cells=80, sparsity=0.05)
    subjects = ["cat", "dog", "bird", "fish", "cat", "dog", "bird", "fish", "cat", "dog"]
    verbs = {"eats", "sings", "swims", "drinks", "chases", "builds", "jumps"}
    svo = 0
    for subject in subjects:
        generated = model.generate([tok.word_to_id[subject]], max_steps=4, temperature=1.0, top_k=3)
        words = [tok.id_to_word.get(t, "") for t in generated]
        if len(words) >= 3 and words[0] in subjects and words[1] in verbs:
            svo += 1
    ok = svo >= 7
    return ok, f"{svo}/10"


def phase3_g34():
    tok, ids, model = train_columns(MEDICAL_GEN_CORPUS, vocab_size=60, epochs=8)
    prefix = [tok.word_to_id["stroke"]]
    generated = model.generate(prefix, max_steps=24, temperature=1.0, top_k=3)

    enc = SemanticSDREncoder(tok.vocab_size, sdr_dim=1024, sparsity=0.02,
                             embed_dim=32, seed=42)
    enc.fit([ids], window=5, verbose=False)
    enc.load_builtin_embeddings(tok.word_to_id)
    topic = enc.encode(tok.word_to_id["stroke"])
    overlaps = []
    for token in generated:
        overlaps.append(float(np.sum(topic * enc.encode(token))) / max(1, enc.k))
    coherence = float(np.mean([ov >= 0.10 for ov in overlaps]))
    ok = coherence >= 0.60
    return ok, f"{coherence:.2f}"


def phase3_g35():
    tok, ids, model = train_columns(GRAMMAR_CORPUS * 3, epochs=6)
    lengths = [20, 40, 80, 120]
    times = []
    for n in lengths:
        model.reset_state()
        sample = (ids * ((n // len(ids)) + 1))[:n]
        t0 = time.perf_counter()
        for token in sample:
            model.forward(token, learn=False)
        times.append((time.perf_counter() - t0) / n)
    ratio = max(times) / max(min(times), 1e-9)
    ok = ratio < 3.0
    return ok, f"max/min={ratio:.2f}"


def phase3_g36():
    domain_a = "cat eats fish . dog eats meat . bird sings song . " * 8
    domain_b = "apple grows tree . orange makes juice . grape grows vine . " * 8
    tok = SimpleWordTokenizer(vocab_size=80)
    tok.build_vocab(domain_a + domain_b)
    ids_a = tok.encode(domain_a)
    ids_b = tok.encode(domain_b)
    model = CorticalColumnSequenceMemory(tok.vocab_size, cells_per_column=80,
                                         sparsity=0.05, seed=42)
    for _ in range(6):
        model.reset_state()
        for token in ids_a:
            model.forward(token, learn=True)
    pairs = [([*ids_a[max(0, i - 4):i + 1]], ids_a[i + 1])
             for i in range(4, len(ids_a) - 1)]
    before = next_accuracy(model, pairs)
    for _ in range(3):
        model.reset_state()
        for token in ids_b:
            model.forward(token, learn=True)
    after = next_accuracy(model, pairs)
    retention = after / max(before, 1e-9)
    ok = retention >= 0.85
    return ok, f"{100*retention:.1f}%"


def structural_scan():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    files = [
        "lsl/model.py", "lsl/synapse.py", "lsl/ssm.py", "lsl/sdr.py",
        "lsl/semantic_sdr.py", "lsl/associative_memory.py",
        "lsl/cortical_column.py", "lsl/__init__.py",
    ]
    forbidden = [
        "to" + "rch", "tensor" + "flow", "ja" + "x", "." + "backward",
        "optimizer" + ".step", "Gradient" + "Tape",
        "B_" + "rec", "B_" + "ssm", "B_" + "emb",
        "dfa" + "_update", "Living" + "Attention" + "Layer",
        "last_" + "attention" + "_map", "attention" + "_map",
        "self-" + "attention", "cross-" + "attention",
    ]
    hits = []
    for rel in files:
        path = os.path.join(root, rel)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for item in forbidden:
            if item in text:
                hits.append(f"{rel}: {item}")
    ok = not hits
    return ok, "clean" if ok else "; ".join(hits)


def main():
    np.random.seed(42)
    print("Strict GOAL.md Benchmark")

    tok, ids, enc, loaded = phase1.build()
    checks = {
        "G1.1": phase1.g11_semantic(enc, tok),
        "G1.2": phase1.g12_capacity(enc),
        "G1.3": phase1.g13_interference(enc, ids),
        "G1.4": phase1.g14_one_shot(enc, ids),
        "G1.5": phase1.g15_completion(enc, ids),
        "G1.6": phase1.g16_sparse_compute(enc),
    }

    pc_tok, pc_ids = phase2.build_corpus()
    ok, val = phase2.test_g21_error_convergence(pc_tok, pc_ids)
    checks["G2.1"] = (ok, f"{100*val:.1f}%")
    ok, val = phase2.test_g22_suppression(pc_tok, pc_ids)
    checks["G2.2"] = (ok, f"{100*val:.1f}%")
    ok, val = phase2.test_g23_local_only(pc_tok)
    checks["G2.3"] = (ok, f"{val} violations")
    ok, val = phase2.test_g24_loss(pc_tok, pc_ids, n_epochs=25)
    checks["G2.4"] = (ok, f"{val:.4f}")
    ok, val = phase2.test_g25_energy(pc_tok, pc_ids)
    checks["G2.5"] = (ok, f"{100*val:.1f}%")
    ok, val = phase2.test_g26_reasoning(pc_tok, pc_ids, n_epochs=25)
    checks["G2.6"] = (ok, f"{val:.3f}")

    checks["G3.1"] = phase3_g31()
    checks["G3.2"] = phase3_g32()
    checks["G3.3"] = phase3_g33()
    checks["G3.4"] = phase3_g34()
    checks["G3.5"] = phase3_g35()
    checks["G3.6"] = phase3_g36()
    checks["Structural"] = structural_scan()

    print("\nSummary")
    passed = 0
    for name, (ok, value) in checks.items():
        passed += int(ok)
        print(f"  {name:10s} {'PASS' if ok else 'FAIL'} {value}")
    goal_passed = sum(1 for name, (ok, _) in checks.items()
                      if name.startswith("G") and ok)
    print(f"\nGoals passed: {goal_passed}/18")
    print(f"Overall: {'PASS' if goal_passed == 18 and checks['Structural'][0] else 'FAIL'}")
    return 0 if goal_passed == 18 and checks["Structural"][0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
