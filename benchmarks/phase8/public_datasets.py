"""Official public dataset downloaders and parsers for Phase 8.

The helpers in this file only download/cache/normalize datasets and implement
gold-answer judges. They are deliberately separate from the strict model path.
"""
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence


USER_AGENT = "LSL-Phase8-PublicBenchmarks/1.0"

BABI_URLS = [
    "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
]
SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}
GSM8K_URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}
MBPP_URLS = {
    "sanitized": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
    "full": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
}


def default_cache_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "public"))


def _download(urls: Sequence[str], path: str, download: bool = True) -> str:
    path = os.path.abspath(path)
    if os.path.exists(path):
        return path
    if not download:
        raise FileNotFoundError(f"Missing cached file: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last_error = None
    tmp = path + ".tmp"
    for url in urls:
        curl = shutil.which("curl") or shutil.which("curl.exe")
        if curl:
            try:
                result = subprocess.run(
                    [curl, "-fL", "--max-time", "240", "--retry", "3", "-A", USER_AGENT, "-o", tmp, url],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    os.replace(tmp, path)
                    return path
                last_error = RuntimeError(result.stderr.strip() or result.stdout.strip())
            except Exception as exc:  # pragma: no cover - depends on local curl/network.
                last_error = exc
                if os.path.exists(tmp):
                    os.remove(tmp)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as response, open(tmp, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, path)
            return path
        except Exception as exc:  # pragma: no cover - depends on remote availability.
            last_error = exc
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(f"Could not download {path}: {last_error}")


def _read_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    rows = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalize_answer(text) -> str:
    value = str(text).lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def exact_match(prediction, answers: Iterable[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(answer) for answer in answers)


def extract_gsm8k_final(answer: str) -> Optional[str]:
    match = re.search(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", str(answer))
    if not match:
        return None
    return match.group(1).replace(",", "")


def numeric_match(prediction, answer, tolerance: float = 1e-6) -> bool:
    try:
        pred = float(str(prediction).replace(",", "").strip())
        gold = float(str(answer).replace(",", "").strip())
    except ValueError:
        return False
    return abs(pred - gold) <= tolerance


def ensure_babi(cache_dir: Optional[str] = None, download: bool = True) -> Dict[str, str]:
    root = os.path.abspath(cache_dir or default_cache_dir())
    babi_dir = os.path.join(root, "babi")
    archive = os.path.join(babi_dir, "babi_tasks_1-20_v1-2.tar.gz")
    extracted = os.path.join(babi_dir, "tasks_1-20_v1-2")
    if not os.path.isdir(extracted):
        _download(BABI_URLS, archive, download=download)
        os.makedirs(babi_dir, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            base = os.path.abspath(babi_dir)
            for member in tar.getmembers():
                target = os.path.abspath(os.path.join(babi_dir, member.name))
                if not (target == base or target.startswith(base + os.sep)):
                    raise RuntimeError(f"Unsafe path in bAbI archive: {member.name}")
            tar.extractall(babi_dir)
    return {"root": extracted, "archive": archive}


def _parse_babi_file(path: str, task_id: int, split: str, limit: Optional[int] = None) -> List[dict]:
    examples = []
    story: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            num_raw, body = line.split(" ", 1)
            if int(num_raw) == 1:
                story = []
            if "\t" in body:
                question, answer, supports = body.split("\t")
                examples.append(
                    {
                        "dataset": "babi",
                        "task_id": task_id,
                        "split": split,
                        "story": list(story),
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "supports": [int(x) for x in supports.split() if x.isdigit()],
                    }
                )
                if limit is not None and len(examples) >= limit:
                    break
            else:
                story.append(body.strip())
    return examples


def load_babi(
    cache_dir: Optional[str] = None,
    split: str = "test",
    language: str = "en",
    tasks: Optional[Sequence[int]] = None,
    limit_per_task: Optional[int] = None,
    download: bool = True,
) -> List[dict]:
    paths = ensure_babi(cache_dir, download=download)
    task_dir = os.path.join(paths["root"], language)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Missing bAbI language directory: {task_dir}")
    rows = []
    for path in sorted(glob(os.path.join(task_dir, f"qa*_{split}.txt"))):
        name = os.path.basename(path)
        match = re.match(r"qa(\d+)_", name)
        if not match:
            continue
        task_id = int(match.group(1))
        if tasks is not None and task_id not in set(int(t) for t in tasks):
            continue
        rows.extend(_parse_babi_file(path, task_id, split, limit=limit_per_task))
    return rows


def ensure_squad(cache_dir: Optional[str] = None, download: bool = True) -> Dict[str, str]:
    root = os.path.abspath(cache_dir or default_cache_dir())
    out_dir = os.path.join(root, "squad_v1_1")
    return {
        split: _download([url], os.path.join(out_dir, f"{split}-v1.1.json"), download=download)
        for split, url in SQUAD_URLS.items()
    }


def load_squad(
    cache_dir: Optional[str] = None,
    split: str = "dev",
    limit: Optional[int] = None,
    download: bool = True,
) -> List[dict]:
    paths = ensure_squad(cache_dir, download=download)
    with open(paths[split], "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for article in raw.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = [a.get("text", "") for a in qa.get("answers", []) if a.get("text")]
                if not answers:
                    continue
                rows.append(
                    {
                        "dataset": "squad_v1_1",
                        "split": split,
                        "title": title,
                        "context": context,
                        "question": qa.get("question", ""),
                        "answers": answers,
                        "id": qa.get("id", ""),
                    }
                )
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def ensure_gsm8k(cache_dir: Optional[str] = None, download: bool = True) -> Dict[str, str]:
    root = os.path.abspath(cache_dir or default_cache_dir())
    out_dir = os.path.join(root, "gsm8k")
    return {
        split: _download([url], os.path.join(out_dir, f"{split}.jsonl"), download=download)
        for split, url in GSM8K_URLS.items()
    }


def load_gsm8k(
    cache_dir: Optional[str] = None,
    split: str = "test",
    limit: Optional[int] = None,
    download: bool = True,
) -> List[dict]:
    paths = ensure_gsm8k(cache_dir, download=download)
    rows = []
    for row in _read_jsonl(paths[split], limit=limit):
        answer = extract_gsm8k_final(row.get("answer", ""))
        if answer is None:
            continue
        rows.append(
            {
                "dataset": "gsm8k",
                "split": split,
                "question": row.get("question", ""),
                "solution": row.get("answer", ""),
                "answer": answer,
            }
        )
    return rows


def ensure_mbpp(cache_dir: Optional[str] = None, download: bool = True) -> Dict[str, str]:
    root = os.path.abspath(cache_dir or default_cache_dir())
    out_dir = os.path.join(root, "mbpp")
    return {
        "sanitized": _download([MBPP_URLS["sanitized"]], os.path.join(out_dir, "sanitized-mbpp.json"), download=download),
        "full": _download([MBPP_URLS["full"]], os.path.join(out_dir, "mbpp.jsonl"), download=download),
    }


def _load_json_or_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_mbpp(
    cache_dir: Optional[str] = None,
    split: str = "sanitized",
    limit: Optional[int] = None,
    download: bool = True,
) -> List[dict]:
    paths = ensure_mbpp(cache_dir, download=download)
    rows = _load_json_or_jsonl(paths[split])
    out = []
    for row in rows:
        task_id = int(row.get("task_id", row.get("id", len(out))))
        tests = row.get("test_list", row.get("tests", []))
        if isinstance(tests, str):
            tests = [tests]
        out.append(
            {
                "dataset": "mbpp",
                "split": split,
                "task_id": task_id,
                "prompt": row.get("prompt", row.get("text", "")),
                "code": row.get("code", ""),
                "tests": [str(test) for test in tests],
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def run_python_tests(code: str, tests: Sequence[str], timeout: float = 3.0) -> bool:
    script = str(code).rstrip() + "\n\n"
    script += "\n".join(str(test) for test in tests)
    with tempfile.TemporaryDirectory(prefix="phase8_mbpp_") as tmpdir:
        path = os.path.join(tmpdir, "candidate.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
            f.write("\n")
        try:
            result = subprocess.run(
                [sys.executable, "-I", path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def dataset_card() -> Dict[str, str]:
    return {
        "babi": BABI_URLS[0],
        "squad_v1_1_train": SQUAD_URLS["train"],
        "squad_v1_1_dev": SQUAD_URLS["dev"],
        "gsm8k_train": GSM8K_URLS["train"],
        "gsm8k_test": GSM8K_URLS["test"],
        "mbpp_sanitized": MBPP_URLS["sanitized"],
        "mbpp_full": MBPP_URLS["full"],
    }
