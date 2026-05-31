"""Run metadata and result persistence helpers for LSL benchmarks."""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit(root: Optional[str] = None) -> str:
    cwd = root or str(project_root())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty(root: Optional[str] = None) -> bool:
    cwd = root or str(project_root())
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_metadata(
    benchmark: str,
    dataset: str,
    seed: Optional[int] = None,
    config: Optional[Dict[str, object]] = None,
    command: Optional[str] = None,
) -> Dict[str, object]:
    root = project_root()
    return {
        "benchmark": str(benchmark),
        "dataset": str(dataset),
        "seed": None if seed is None else int(seed),
        "timestamp": now_utc(),
        "git_commit": git_commit(str(root)),
        "git_dirty": git_dirty(str(root)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": command or " ".join(sys.argv),
        "config": dict(config or {}),
    }


def default_result_path(
    benchmark: str,
    dataset: str,
    seed: Optional[int] = None,
    results_root: str = "results",
    suffix: str = ".json",
) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    commit = git_commit()
    seed_part = "noseed" if seed is None else f"seed{int(seed)}"
    safe_benchmark = _safe_name(benchmark)
    safe_dataset = _safe_name(dataset)
    name = f"{stamp}_{commit}_{seed_part}{suffix}"
    return project_root() / results_root / safe_benchmark / safe_dataset / name


def write_result(
    payload: Dict[str, object],
    benchmark: str,
    dataset: str,
    seed: Optional[int] = None,
    config: Optional[Dict[str, object]] = None,
    output_path: Optional[str] = None,
    results_root: str = "results",
) -> str:
    out = Path(output_path) if output_path else default_result_path(benchmark, dataset, seed, results_root)
    if not out.is_absolute():
        out = project_root() / out
    full_payload = dict(payload)
    full_payload.setdefault("metadata", run_metadata(benchmark, dataset, seed=seed, config=config))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2, ensure_ascii=False)
    _append_index(out, full_payload, results_root=results_root)
    return str(out)


def _append_index(path: Path, payload: Dict[str, object], results_root: str = "results") -> None:
    root = project_root() / results_root
    root.mkdir(parents=True, exist_ok=True)
    meta = payload.get("metadata", {})
    row = {
        "path": str(path),
        "benchmark": meta.get("benchmark"),
        "dataset": meta.get("dataset"),
        "seed": meta.get("seed"),
        "timestamp": meta.get("timestamp"),
        "git_commit": meta.get("git_commit"),
        "success": payload.get("success"),
    }
    with open(root / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_name(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip().lower()]
    out = "".join(chars).strip("_")
    return out or "run"

