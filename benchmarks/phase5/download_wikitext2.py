"""Download/cache WikiText-2 raw text from Hugging Face parquet files."""
import argparse
import os
import urllib.request
from typing import Dict

import pyarrow.parquet as pq


REPO_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"
FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def default_cache_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wikitext-2-raw-v1"))


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _parquet_to_text(parquet_path: str, text_path: str) -> None:
    table = pq.read_table(parquet_path)
    rows = table.column("text").to_pylist()
    lines = [str(row) for row in rows if str(row).strip()]
    with open(text_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")


def ensure_wikitext2(cache_dir: str = None, download: bool = True) -> Dict[str, str]:
    cache_dir = os.path.abspath(cache_dir or default_cache_dir())
    os.makedirs(cache_dir, exist_ok=True)
    result = {}
    for split, filename in FILES.items():
        parquet_path = os.path.join(cache_dir, filename)
        text_path = os.path.join(cache_dir, f"wiki.{split}.raw.txt")
        if not os.path.exists(text_path):
            if not os.path.exists(parquet_path):
                if not download:
                    raise FileNotFoundError(f"Missing {text_path}; rerun with download enabled")
                _download(f"{REPO_BASE}/{filename}", parquet_path)
            _parquet_to_text(parquet_path, text_path)
        result[split] = text_path
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    paths = ensure_wikitext2(args.cache_dir, download=not args.no_download)
    print("WikiText-2 raw cache ready:")
    for split, path in paths.items():
        print(f"  {split:<10} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
