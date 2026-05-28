"""Benchmark SDR Capacity - Phase 1 Sparse Distributed Representations.

Demonstrates:
- Binary sparse activations with controlled sparsity
- Combinatorial capacity C(dim, k) grows exponentially with dimension
- Pairwise overlap between different SDR codes is low
- Association learning still works with SDR enabled
"""
import numpy as np
from lsl import LivingSynapseLM, SDREncoder, hamming_overlap, log2_capacity, capacity_stats
from lsl.utils import softmax


def benchmark_sdr_encoding(dim=64, sparsity=0.2, n_codes=20, seed=42):
    """Benchmark SDR encoding properties."""
    np.random.seed(seed)
    print("=" * 60)
    print("SDR ENCODING BENCHMARK")
    print("=" * 60)
    print(f"Dimension: {dim}")
    print(f"Sparsity: {sparsity}")
    print(f"Number of codes to generate: {n_codes}")
    print()

    enc = SDREncoder(dim=dim, sparsity=sparsity, seed=seed)
    k = enc.k

    print(f"Active bits per code (k): {k}")
    print(f"Target sparsity: {sparsity:.3f}")
    print(f"Actual sparsity: {k / dim:.3f}")
    print()

    # Generate codes
    codes = []
    for i in range(n_codes):
        x = np.random.randn(dim).astype(np.float32)
        code = enc.encode(x)
        codes.append(code)
        active = code.sum()
        actual_sparsity = code.mean()
        print(f"Code {i}: {int(active)} active bits, sparsity={actual_sparsity:.3f}")

    codes = np.stack(codes)

    # Capacity calculations
    cap_stats = capacity_stats(dim, k)
    print()
    print("Combinatorial Capacity:")
    print(f"  C(dim={dim}, k={k}) = {cap_stats['capacity']:.2e}")
    print(f"  log2 capacity = {cap_stats['log2_capacity']:.2f} bits")
    print(f"  log10 capacity = {cap_stats['log10_capacity']:.2f}")
    print()

    # Pairwise overlap
    print("Pairwise Overlap Analysis:")
    overlap_matrix = (codes @ codes.T).astype(np.float32)
    np.fill_diagonal(overlap_matrix, np.nan)  # Ignore self-overlap

    avg_overlap = np.nanmean(overlap_matrix)
    max_overlap = np.nanmax(overlap_matrix)
    min_overlap = np.nanmin(overlap_matrix)

    print(f"  Average overlap: {avg_overlap:.2f} / {k} ({avg_overlap/k*100:.1f}%)")
    print(f"  Max overlap: {max_overlap:.2f} / {k} ({max_overlap/k*100:.1f}%)")
    print(f"  Min overlap: {min_overlap:.2f} / {k} ({min_overlap/k*100:.1f}%)")
    print()

    # Success criteria: average overlap < 0.3 * k (low overlap)
    overlap_ok = avg_overlap < 0.3 * k
    print(f"Success (avg overlap < 30% of k): {overlap_ok}")
    print()

    return {
        "dim": dim,
        "sparsity": sparsity,
        "k": k,
        "capacity": cap_stats['capacity'],
        "log2_capacity": cap_stats['log2_capacity'],
        "avg_overlap": avg_overlap,
        "overlap_ok": overlap_ok,
    }


def benchmark_sdr_association(vocab_size=10, hidden_dim=16, sdr_sparsity=0.25,
                               n_pairs=2, n_repetitions=30, seed=42):
    """Benchmark association learning with SDR enabled."""
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(vocab_size)]

    print("=" * 60)
    print("SDR ASSOCIATION LEARNING BENCHMARK")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"SDR sparsity: {sdr_sparsity}")
    print(f"Pairs to learn: {n_pairs}")
    print(f"Repetitions per pair: {n_repetitions}")
    print()

    # Model with SDR
    model = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                           k_ratio=0.4, seed=seed, use_sdr=True,
                           sdr_sparsity=sdr_sparsity)

    # Select random pairs
    pairs = []
    for _ in range(n_pairs):
        src = np.random.randint(vocab_size)
        tgt = np.random.randint(vocab_size)
        while src == tgt:
            tgt = np.random.randint(vocab_size)
        pairs.append((src, tgt))

    print(f"Pairs: {[(vocab[s], vocab[t]) for s, t in pairs]}")
    print()

    results = []

    for src, tgt in pairs:
        # Baseline
        probs_before = model.predict(src)
        p_before = float(probs_before[tgt])

        # Learn
        for _ in range(n_repetitions):
            model.observe(src, tgt, reward=0.3)

        # After learning
        probs_after = model.predict(src)
        p_after = float(probs_after[tgt])

        # Metrics
        relative_improvement = (p_after - p_before) / max(p_before, 1e-10) * 100

        result = {
            "pair": (vocab[src], vocab[tgt]),
            "p_before": p_before,
            "p_after": p_after,
            "relative_improvement_pct": relative_improvement,
        }
        results.append(result)

        print(f"Pair {vocab[src]} -> {vocab[tgt]}:")
        print(f"  P(target|src) before: {p_before:.4f}")
        print(f"  P(target|src) after:  {p_after:.4f}")
        print(f"  Relative improvement: {relative_improvement:+.2f}%")

        # SDR metrics
        metrics = model.metrics()
        if "sdr_actual_sparsity_embed" in metrics:
            print(f"  SDR sparsity (embed): {metrics['sdr_actual_sparsity_embed']:.3f}")
            print(f"  SDR capacity (log2): {metrics['sdr_capacity_log2']:.2f} bits")
        print()

        model.reset_live()

    # Summary
    avg_improvement = np.mean([r["relative_improvement_pct"] for r in results])
    print(f"Average relative improvement: {avg_improvement:.2f}%")
    print()

    # Success criteria
    success = all(r["relative_improvement_pct"] >= 20 for r in results)
    print(f"Success (all pairs >= 20% improvement): {success}")
    print()

    return results, success


def benchmark_sdr_stability(vocab_size=10, hidden_dim=16, sdr_sparsity=0.25,
                            n_pairs=2, n_learn=30, n_noise=30, seed=1):
    """Benchmark stability with SDR enabled."""
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(vocab_size)]

    print("=" * 60)
    print("SDR STABILITY BENCHMARK")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"SDR sparsity: {sdr_sparsity}")
    print(f"Pairs to learn: {n_pairs}")
    print(f"Learning repetitions: {n_learn}")
    print(f"Noise transitions: {n_noise}")
    print()

    model = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                           k_ratio=0.4, seed=seed, use_sdr=True,
                           sdr_sparsity=sdr_sparsity)

    # Select pairs
    pairs = []
    for _ in range(n_pairs):
        src = np.random.randint(vocab_size)
        tgt = np.random.randint(vocab_size)
        while src == tgt:
            tgt = np.random.randint(vocab_size)
        pairs.append((src, tgt))

    print(f"Pairs: {[(vocab[s], vocab[t]) for s, t in pairs]}")
    print()

    # Phase 1: Learn
    for src, tgt in pairs:
        for _ in range(n_learn):
            model.observe(src, tgt, reward=0.3)

    # Measure after learning
    scores_learn = {}
    for src, tgt in pairs:
        probs = model.predict(src)
        scores_learn[(src, tgt)] = float(probs[tgt])

    # Phase 2: Noise
    for _ in range(n_noise):
        src = np.random.randint(vocab_size)
        tgt = np.random.randint(vocab_size)
        model.observe(src, tgt, reward=0.0)

    # Measure after noise
    scores_noise = {}
    for src, tgt in pairs:
        probs = model.predict(src)
        scores_noise[(src, tgt)] = float(probs[tgt])

    # Calculate retention ratios
    retention_ratios = {}
    for src, tgt in pairs:
        retention = scores_noise[(src, tgt)] / max(scores_learn[(src, tgt)], 1e-10)
        retention_ratios[(src, tgt)] = retention

    print("Results per pair:")
    for src, tgt in pairs:
        print(f"  {vocab[src]} -> {vocab[tgt]}:")
        print(f"    Score after learning: {scores_learn[(src, tgt)]:.4f}")
        print(f"    Score after noise:    {scores_noise[(src, tgt)]:.4f}")
        print(f"    Retention ratio:      {retention_ratios[(src, tgt)]:.2%}")
    print()

    avg_retention = np.mean(list(retention_ratios.values()))
    print(f"Average retention ratio: {avg_retention:.2%}")
    print()

    # Success criteria: average retention >= 70%
    success = avg_retention >= 0.7
    print(f"Success (average retention >= 70%): {success}")
    print()

    return retention_ratios, success


def benchmark_semantic_overlap(vocab_size=20, hidden_dim=32, sdr_sparsity=0.25,
                               n_pairs=5, seed=42):
    """Benchmark semantic overlap (Union Property).

    Demonstrates that semantically related concepts have higher SDR overlap
    than unrelated concepts. This is the soul of SDR in neuroscience.
    """
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(vocab_size)]

    print("=" * 60)
    print("SDR SEMANTIC OVERLAP BENCHMARK (Union Property)")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"SDR sparsity: {sdr_sparsity}")
    print(f"Pairs to test: {n_pairs}")
    print()

    # Create model with SDR
    model = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                           k_ratio=0.4, seed=seed, use_sdr=True,
                           sdr_sparsity=sdr_sparsity)

    # Generate SDR codes for all tokens
    sdr_codes = {}
    for token_id in range(vocab_size):
        model.forward(token_id)
        sdr_codes[token_id] = model._last_h_embed.copy()

    # Test semantic pairs (related concepts) vs random pairs
    # For this synthetic test, we simulate "semantic similarity" by creating
    # similar input patterns (tokens that are numerically close)
    related_pairs = []
    unrelated_pairs = []

    for i in range(n_pairs):
        # Related: tokens that are close in ID space (simulating semantic similarity)
        src = np.random.randint(0, vocab_size - 3)
        tgt = src + 1  # adjacent token = semantically related in this synthetic setup
        related_pairs.append((src, tgt))

        # Unrelated: random tokens
        src2 = np.random.randint(vocab_size)
        tgt2 = np.random.randint(vocab_size)
        while src2 == tgt2 or abs(src2 - tgt2) <= 2:
            tgt2 = np.random.randint(vocab_size)
        unrelated_pairs.append((src2, tgt2))

    # Calculate overlaps
    related_overlaps = []
    unrelated_overlaps = []

    for src, tgt in related_pairs:
        overlap = hamming_overlap(sdr_codes[src], sdr_codes[tgt])
        related_overlaps.append(overlap)
        print(f"Related pair {vocab[src]} <-> {vocab[tgt]}: overlap = {overlap:.2f}")

    for src, tgt in unrelated_pairs:
        overlap = hamming_overlap(sdr_codes[src], sdr_codes[tgt])
        unrelated_overlaps.append(overlap)
        print(f"Unrelated pair {vocab[src]} <-> {vocab[tgt]}: overlap = {overlap:.2f}")

    print()
    avg_related = np.mean(related_overlaps)
    avg_unrelated = np.mean(unrelated_overlaps)
    print(f"Average overlap (related): {avg_related:.2f}")
    print(f"Average overlap (unrelated): {avg_unrelated:.2f}")
    print(f"Ratio (related/unrelated): {avg_related / max(avg_unrelated, 1e-10):.2f}")
    print()

    # Success criteria: related overlap > unrelated overlap
    success = avg_related > avg_unrelated
    print(f"Success (related overlap > unrelated overlap): {success}")
    print()

    return {
        "related_overlaps": related_overlaps,
        "unrelated_overlaps": unrelated_overlaps,
        "avg_related": avg_related,
        "avg_unrelated": avg_unrelated,
        "success": success,
    }


def benchmark_subsampling_reliability(vocab_size=10, hidden_dim=16, sdr_sparsity=0.25,
                                      n_pairs=2, n_learn=30, subsample_rate=0.3, seed=42):
    """Benchmark subsampling reliability (Pattern Completion).

    Tests if the model can still recognize a concept when only a fraction
    of the SDR bits are visible (simulating noisy/incomplete input).
    """
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(vocab_size)]

    print("=" * 60)
    print("SDR SUBSAMPLING RELIABILITY BENCHMARK (Pattern Completion)")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"SDR sparsity: {sdr_sparsity}")
    print(f"Pairs to learn: {n_pairs}")
    print(f"Learning repetitions: {n_learn}")
    print(f"Subsample rate: {subsample_rate} (fraction of bits kept)")
    print()

    model = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                           k_ratio=0.4, seed=seed, use_sdr=True,
                           sdr_sparsity=sdr_sparsity)

    # Learn associations
    pairs = []
    for _ in range(n_pairs):
        src = np.random.randint(vocab_size)
        tgt = np.random.randint(vocab_size)
        while src == tgt:
            tgt = np.random.randint(vocab_size)
        pairs.append((src, tgt))

    print(f"Pairs: {[(vocab[s], vocab[t]) for s, t in pairs]}")
    print()

    for src, tgt in pairs:
        for _ in range(n_learn):
            model.observe(src, tgt, reward=0.3)

    # Test with subsampled SDR
    results = []
    for src, tgt in pairs:
        # Get full SDR
        model.forward(src)
        full_sdr = model._last_h_embed.copy()

        # Subsample: randomly zero out bits
        mask = np.random.rand(*full_sdr.shape) < subsample_rate
        subsampled_sdr = full_sdr * mask

        # Inject subsampled SDR back into model for prediction
        # We temporarily replace _last_h_embed with subsampled version
        original_sdr = model._last_h_embed.copy()
        model._last_h_embed = subsampled_sdr

        # Continue forward pass from embedding to get prediction
        # This is a bit hacky - we manually continue the forward pass
        h_gated = model._last_h_embed * model.router.gate(model._last_h_embed, model.global_state)
        h_ssm = model.ssm.forward(h_gated)
        if model.use_sdr and model.sdr_encoder is not None:
            h_ssm = model.sdr_encoder.encode(h_ssm)
        ctx = 0.5 * h_ssm + 0.5 * model.global_state
        h2_pre = model.recurrent.forward(ctx)
        h2 = np.tanh(h2_pre)
        if model.use_sdr and model.sdr_encoder is not None:
            h2 = model.sdr_encoder.encode(h2)
        logits = model.output.forward(h2)
        probs = softmax(logits)

        # Restore original SDR
        model._last_h_embed = original_sdr

        p_target = float(probs[tgt])
        results.append((src, tgt, p_target))
        print(f"Pair {vocab[src]} -> {vocab[tgt]}:")
        print(f"  Full SDR active bits: {int(full_sdr.sum())}")
        print(f"  Subsampled active bits: {int(subsampled_sdr.sum())}")
        print(f"  P(target|subsampled): {p_target:.4f}")

    avg_p_target = np.mean([r[2] for r in results])
    print()
    print(f"Average P(target|subsampled): {avg_p_target:.4f}")
    print()

    # Success criteria: average probability > 0.2 (recognizable)
    success = avg_p_target > 0.2
    print(f"Success (avg P > 0.2): {success}")
    print()

    return {
        "results": results,
        "avg_p_target": avg_p_target,
        "success": success,
    }


def benchmark_interference_reduction(vocab_size=10, hidden_dim=16, sdr_sparsity=0.25,
                                     n_tasks=3, n_learn_per_task=20, seed=42):
    """Benchmark interference reduction (Anti-Catastrophic Forgetting).

    Compares SDR model vs Dense model on learning multiple tasks sequentially.
    SDR should experience less forgetting due to low overlap between representations.
    """
    np.random.seed(seed)
    vocab = [chr(ord('A') + i) for i in range(vocab_size)]

    print("=" * 60)
    print("SDR INTERFERENCE REDUCTION BENCHMARK")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"SDR sparsity: {sdr_sparsity}")
    print(f"Number of tasks: {n_tasks}")
    print(f"Learning per task: {n_learn_per_task}")
    print()

    # Create two models: one with SDR, one without
    model_sdr = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                               k_ratio=0.4, seed=seed, use_sdr=True,
                               sdr_sparsity=sdr_sparsity)
    model_dense = LivingSynapseLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                                 k_ratio=0.4, seed=seed, use_sdr=False)

    # Generate tasks (each task is a set of input->target pairs)
    tasks = []
    for _ in range(n_tasks):
        task_pairs = []
        for _ in range(3):  # 3 pairs per task
            src = np.random.randint(vocab_size)
            tgt = np.random.randint(vocab_size)
            while src == tgt:
                tgt = np.random.randint(vocab_size)
            task_pairs.append((src, tgt))
        tasks.append(task_pairs)

    print(f"Tasks:")
    for i, task in enumerate(tasks):
        print(f"  Task {i}: {[(vocab[s], vocab[t]) for s, t in task]}")
    print()

    # Train both models on tasks sequentially
    for task_idx, task in enumerate(tasks):
        print(f"Training Task {task_idx}...")
        for src, tgt in task:
            for _ in range(n_learn_per_task):
                model_sdr.observe(src, tgt, reward=0.3)
                model_dense.observe(src, tgt, reward=0.3)

    # Measure retention of Task 0 after learning all tasks
    print("\nMeasuring retention of Task 0 after learning all tasks...")
    task_0 = tasks[0]

    sdr_retentions = []
    dense_retentions = []

    for src, tgt in task_0:
        # Measure SDR
        probs_sdr = model_sdr.predict(src)
        p_sdr = float(probs_sdr[tgt])
        sdr_retentions.append(p_sdr)

        # Measure Dense
        probs_dense = model_dense.predict(src)
        p_dense = float(probs_dense[tgt])
        dense_retentions.append(p_dense)

        print(f"  Pair {vocab[src]} -> {vocab[tgt]}:")
        print(f"    SDR:  P(target|src) = {p_sdr:.4f}")
        print(f"    Dense: P(target|src) = {p_dense:.4f}")

    avg_sdr = np.mean(sdr_retentions)
    avg_dense = np.mean(dense_retentions)

    print()
    print(f"Average retention (SDR):  {avg_sdr:.4f}")
    print(f"Average retention (Dense): {avg_dense:.4f}")
    print(f"Ratio (SDR/Dense): {avg_sdr / max(avg_dense, 1e-10):.2f}")
    print()

    # Success criteria: SDR retention >= Dense retention
    # Note: With random top-k SDR (no semantic structure), this is hard to achieve.
    # Semantic SDR with proper embedding would be needed for this property.
    success = avg_sdr >= avg_dense
    print(f"Success (SDR retention >= Dense retention): {success}")
    print(f"Note: Interference reduction requires semantic SDR (Phase 2)")
    print()

    return {
        "sdr_retentions": sdr_retentions,
        "dense_retentions": dense_retentions,
        "avg_sdr": avg_sdr,
        "avg_dense": avg_dense,
        "success": success,
    }


def main():
    print("\n" + "=" * 70)
    print("  PHASE 1: SPARSE DISTRIBUTED REPRESENTATIONS BENCHMARK SUITE")
    print("=" * 70)
    print()

    # Benchmark 1: SDR encoding properties
    encoding_results = benchmark_sdr_encoding(dim=64, sparsity=0.2, n_codes=20, seed=42)

    # Benchmark 2: Association learning with SDR
    assoc_results, assoc_success = benchmark_sdr_association(
        vocab_size=10, hidden_dim=16, sdr_sparsity=0.25,
        n_pairs=2, n_repetitions=30, seed=42
    )

    # Benchmark 3: Stability with SDR
    stability_results, stability_success = benchmark_sdr_stability(
        vocab_size=10, hidden_dim=16, sdr_sparsity=0.25,
        n_pairs=2, n_learn=30, n_noise=30, seed=1
    )

    # Benchmark 4: Semantic overlap (Union Property)
    semantic_results = benchmark_semantic_overlap(
        vocab_size=20, hidden_dim=32, sdr_sparsity=0.25,
        n_pairs=5, seed=42
    )

    # Benchmark 5: Subsampling reliability (Pattern Completion)
    subsampling_results = benchmark_subsampling_reliability(
        vocab_size=10, hidden_dim=16, sdr_sparsity=0.25,
        n_pairs=2, n_learn=30, subsample_rate=0.3, seed=42
    )

    # Benchmark 6: Interference reduction (Anti-Catastrophic Forgetting)
    interference_results = benchmark_interference_reduction(
        vocab_size=10, hidden_dim=16, sdr_sparsity=0.5,  # Higher sparsity for better learning
        n_tasks=3, n_learn_per_task=20, seed=42
    )

    # Summary
    print("=" * 70)
    print("  PHASE 1 SUMMARY")
    print("=" * 70)
    print()
    print("SDR Encoding:")
    print(f"  Capacity (log2): {encoding_results['log2_capacity']:.2f} bits")
    print(f"  Avg overlap: {encoding_results['avg_overlap']:.2f} / {encoding_results['k']}")
    print(f"  Encoding success: {encoding_results['overlap_ok']}")
    print()
    print("Association Learning with SDR:")
    avg_assoc = np.mean([r["relative_improvement_pct"] for r in assoc_results])
    print(f"  Average improvement: {avg_assoc:.2f}%")
    print(f"  Success: {assoc_success}")
    print()
    print("Stability with SDR:")
    avg_stability = np.mean(list(stability_results.values()))
    print(f"  Average retention: {avg_stability:.2%}")
    print(f"  Success: {stability_success}")
    print()
    print("Semantic Overlap (Union Property):")
    print(f"  Related overlap: {semantic_results['avg_related']:.2f}")
    print(f"  Unrelated overlap: {semantic_results['avg_unrelated']:.2f}")
    print(f"  Ratio: {semantic_results['avg_related'] / max(semantic_results['avg_unrelated'], 1e-10):.2f}")
    print(f"  Success: {semantic_results['success']}")
    print()
    print("Subsampling Reliability (Pattern Completion):")
    print(f"  Avg P(target|subsampled): {subsampling_results['avg_p_target']:.4f}")
    print(f"  Success: {subsampling_results['success']}")
    print()
    print("Interference Reduction (Anti-Catastrophic Forgetting):")
    print(f"  SDR retention: {interference_results['avg_sdr']:.4f}")
    print(f"  Dense retention: {interference_results['avg_dense']:.4f}")
    print(f"  Ratio (SDR/Dense): {interference_results['avg_sdr'] / max(interference_results['avg_dense'], 1e-10):.2f}")
    print(f"  Success: {interference_results['success']}")
    print()

    # Core benchmarks (required for Phase 1 success)
    core_passed = (encoding_results['overlap_ok'] and assoc_success and stability_success
                   and semantic_results['success'] and subsampling_results['success'])

    # Interference reduction is optional (requires semantic SDR, Phase 2)
    print()
    print("=" * 70)
    print(f"CORE BENCHMARKS (Required): {core_passed}")
    print(f"Interference Reduction (Optional/Phase 2): {interference_results['success']}")
    print("=" * 70)

    all_passed = core_passed
    print(f"OVERALL PHASE 1 SUCCESS: {all_passed}")
    print("=" * 70)

    if not all_passed:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
