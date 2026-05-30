"""Download/cache official public datasets used by Phase 8."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.phase8.public_datasets import (
    dataset_card,
    ensure_babi,
    ensure_gsm8k,
    ensure_mbpp,
    ensure_squad,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else None
    download = not args.no_download
    print("Phase 8 public dataset cache")
    print("=" * 88)
    print("Sources:")
    for name, url in dataset_card().items():
        print(f"  {name:<20} {url}")
    print("-" * 88)
    paths = {
        "babi": ensure_babi(cache_dir, download=download),
        "squad": ensure_squad(cache_dir, download=download),
        "gsm8k": ensure_gsm8k(cache_dir, download=download),
        "mbpp": ensure_mbpp(cache_dir, download=download),
    }
    for name, mapping in paths.items():
        print(name)
        for key, path in mapping.items():
            print(f"  {key:<12} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
