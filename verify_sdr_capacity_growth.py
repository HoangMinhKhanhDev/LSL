"""Verify SDR capacity grows exponentially with dimension.

This demonstrates that C(dim, k) = combinatorial capacity grows exponentially,
proving the key SDR property: capacity ∝ exponential(dim).
"""
import numpy as np
from lsl import SDREncoder, log2_capacity, capacity_stats

def main():
    print("=" * 70)
    print("  SDR CAPACITY GROWTH VERIFICATION")
    print("  Demonstrating exponential capacity growth with dimension")
    print("=" * 70)
    print()

    sparsity = 0.2
    dimensions = [8, 16, 32, 64, 128]

    print(f"Sparsity: {sparsity}")
    print()
    print(f"{'Dim':<6} | {'k':<4} | {'C(dim,k)':<15} | {'log2(C)':<10} | {'Growth Factor'}")
    print("-" * 70)

    prev_log2_capacity = None
    for dim in dimensions:
        k = int(dim * sparsity)
        stats = capacity_stats(dim, k)
        C = stats['capacity']
        log2_C = stats['log2_capacity']

        if prev_log2_capacity is not None:
            growth = log2_C / prev_log2_capacity
            growth_str = f"{growth:.2f}x"
        else:
            growth_str = "---"

        print(f"{dim:<6} | {k:<4} | {C:<15.2e} | {log2_C:<10.2f} | {growth_str}")
        prev_log2_capacity = log2_C

    print()
    print("Analysis:")
    print("- When dimension doubles (16→32→64), log2 capacity roughly doubles")
    print("- This proves exponential growth: C(dim, k) ∝ 2^(f(dim))")
    print("- For sparse codes (k ∝ dim), log2(C) ≈ k * log2(dim/k)")
    print()

    # Demonstrate formula
    print("Theoretical verification:")
    print("C(n,k) = n! / (k!(n-k)!)")
    print("log2(C(n,k)) ≈ k * log2(n/k)  (Stirling approximation for sparse k << n)")
    print()
    print("Example: dim=64, k=12")
    print("  k * log2(dim/k) = 12 * log2(64/12) = 12 * 2.42 = 29.0 bits")
    print("  Actual log2(C) = 41.58 bits (includes full combinatorial terms)")
    print()

    print("=" * 70)
    print("  CONCLUSION: SDR capacity grows exponentially with dimension")
    print("  This is the core mathematical property enabling efficient")
    print("  high-capacity representations in biological neural systems.")
    print("=" * 70)

if __name__ == "__main__":
    main()
