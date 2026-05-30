"""Download/cache a larger TinyStories validation text file."""
import argparse
import os
import urllib.request


URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


def default_cache_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tinystories"))


def ensure_tinystories_valid(cache_dir: str = None, download: bool = True) -> str:
    cache_dir = os.path.abspath(cache_dir or default_cache_dir())
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "TinyStoriesV2-GPT4-valid.txt")
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(f"Missing {path}; rerun with download enabled")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as response, open(path, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    path = ensure_tinystories_valid(args.cache_dir, download=not args.no_download)
    print(f"TinyStories validation cache ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
