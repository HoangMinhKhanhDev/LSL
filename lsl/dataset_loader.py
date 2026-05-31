"""Standard dataset loading utilities for LSL training and benchmarks.

The loader keeps corpus handling out of benchmark scripts. It supports named
datasets, train/validation/test splits, streaming line iteration, deterministic
shuffling, token-budget extraction, and small built-in corpora for smoke runs.
"""
from __future__ import annotations

import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetSource:
    name: str
    splits: Dict[str, str]
    language: str = "en"
    description: str = ""
    license: str = "unknown"
    citation: str = ""
    normalize_unicode: bool = True
    vietnamese_normalization: bool = False


@dataclass
class DatasetConfig:
    name: str
    split: str = "train"
    language: Optional[str] = None
    encoding: str = "utf-8"
    shuffle: bool = False
    seed: int = 42
    max_chars: Optional[int] = None
    normalize_unicode: Optional[bool] = None
    vietnamese_normalization: Optional[bool] = None
    repeat: bool = False


@dataclass
class DatasetStats:
    name: str
    split: str
    path: str
    language: str
    total_chars: int = 0
    total_lines: int = 0
    total_words: int = 0
    avg_line_length: float = 0.0
    min_line_length: int = 0
    max_line_length: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "split": self.split,
            "path": self.path,
            "language": self.language,
            "total_chars": int(self.total_chars),
            "total_lines": int(self.total_lines),
            "total_words": int(self.total_words),
            "avg_line_length": float(self.avg_line_length),
            "min_line_length": int(self.min_line_length),
            "max_line_length": int(self.max_line_length),
        }


@dataclass
class DatasetTextSplits:
    dataset: str
    train: str
    validation: str
    test: str
    train_path: str
    validation_path: str
    test_path: str
    language: str

    def metadata(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "language": self.language,
            "train_chars": len(self.train),
            "validation_chars": len(self.validation),
            "test_chars": len(self.test),
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "test_path": self.test_path,
        }


class DatasetLoader:
    """Named corpus loader for reproducible LSL experiments."""

    DATASETS: Dict[str, DatasetSource] = {
        "tinystories": DatasetSource(
            name="tinystories",
            splits={
                "train": "benchmarks/data/tinystories/TinyStoriesV2-GPT4-valid.txt",
                "validation": "benchmarks/data/tinystories/TinyStoriesV2-GPT4-valid.txt",
                "test": "benchmarks/data/tinystories/TinyStoriesV2-GPT4-valid.txt",
                "fallback": "benchmarks/phase4/tinystories_subset.txt",
            },
            language="en",
            description="TinyStories validation text when available; phase4 subset fallback.",
        ),
        "wikitext2": DatasetSource(
            name="wikitext2",
            splits={
                "train": "benchmarks/data/wikitext-2-raw-v1/wiki.train.raw.txt",
                "validation": "benchmarks/data/wikitext-2-raw-v1/wiki.validation.raw.txt",
                "val": "benchmarks/data/wikitext-2-raw-v1/wiki.validation.raw.txt",
                "test": "benchmarks/data/wikitext-2-raw-v1/wiki.test.raw.txt",
            },
            language="en",
            description="WikiText-2 raw train/validation/test splits.",
        ),
        "vietnamese_small": DatasetSource(
            name="vietnamese_small",
            splits={
                "train": "benchmarks/data/vietnamese_small/vietnamese_mini_corpus.txt",
                "validation": "benchmarks/data/vietnamese_small/vietnamese_mini_corpus.txt",
                "test": "benchmarks/data/vietnamese_small/vietnamese_mini_corpus.txt",
            },
            language="vi",
            description="Small Vietnamese seed corpus for pipeline and tokenizer smoke tests.",
            license="project-curated",
            vietnamese_normalization=True,
        ),
        "dialogue_small": DatasetSource(
            name="dialogue_small",
            splits={
                "train": "benchmarks/data/dialogue_small/dialogue_mini_corpus.txt",
                "validation": "benchmarks/data/dialogue_small/dialogue_mini_corpus.txt",
                "test": "benchmarks/data/dialogue_small/dialogue_mini_corpus.txt",
            },
            language="en",
            description="Small dialogue seed corpus for real chat pipeline smoke tests.",
            license="project-curated",
        ),
    }

    ALIASES: Dict[str, Tuple[str, str]] = {
        "tinystories_subset": ("tinystories", "train"),
        "wikitext2_train": ("wikitext2", "train"),
        "wikitext2_val": ("wikitext2", "validation"),
        "wikitext2_validation": ("wikitext2", "validation"),
        "wikitext2_test": ("wikitext2", "test"),
        "vietnamese": ("vietnamese_small", "train"),
        "dialogue": ("dialogue_small", "train"),
    }

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]
        self.stats: Dict[Tuple[str, str], DatasetStats] = {}

    def _source_and_split(self, name: str, split: str = "train") -> Tuple[Optional[DatasetSource], str, str]:
        name = str(name)
        split = str(split or "train").lower()
        if name in self.ALIASES:
            dataset, alias_split = self.ALIASES[name]
            return self.DATASETS[dataset], dataset, alias_split
        source = self.DATASETS.get(name)
        return source, name, split

    def dataset_metadata(self, name: str) -> Dict[str, object]:
        source, dataset, _ = self._source_and_split(name)
        if source is None:
            path = self.resolve_path(name)
            return {
                "name": dataset,
                "language": "unknown",
                "description": "custom path",
                "path": str(path),
                "splits": {"train": str(path)},
            }
        return {
            "name": source.name,
            "language": source.language,
            "description": source.description,
            "license": source.license,
            "citation": source.citation,
            "splits": dict(source.splits),
        }

    def resolve_path(self, name: str, split: str = "train") -> Path:
        source, dataset, split = self._source_and_split(name, split)
        if source is None:
            path = Path(dataset)
            if not path.is_absolute():
                path = self.root_dir / path
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path}")
            return path

        split_path = source.splits.get(split) or source.splits.get("train")
        if split_path and not (self.root_dir / split_path).exists():
            fallback = source.splits.get("fallback")
            if fallback:
                split_path = fallback
        if not split_path:
            raise FileNotFoundError(f"Dataset split not registered: {dataset}:{split}")
        path = self.root_dir / split_path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    def _config_for(self, config: DatasetConfig) -> Tuple[DatasetSource, str]:
        source, dataset, split = self._source_and_split(config.name, config.split)
        if source is None:
            source = DatasetSource(dataset, {"train": str(self.resolve_path(dataset))})
        return source, split

    def normalize_text(self, text: str, config: DatasetConfig) -> str:
        source, _ = self._config_for(config)
        normalize_unicode = source.normalize_unicode if config.normalize_unicode is None else bool(config.normalize_unicode)
        vietnamese_normalization = (
            source.vietnamese_normalization
            if config.vietnamese_normalization is None
            else bool(config.vietnamese_normalization)
        )
        if normalize_unicode:
            text = unicodedata.normalize("NFC", str(text))
        if vietnamese_normalization:
            text = re.sub(r"\s+([.,!?;:)\]])", r"\1", text)
            text = re.sub(r"([(])\s+", r"\1", text)
            text = re.sub(r"\s+", " ", text)
        return text

    def iter_lines(self, config: DatasetConfig) -> Iterator[str]:
        path = self.resolve_path(config.name, config.split)
        max_chars = None if config.max_chars is None else max(0, int(config.max_chars))
        emitted = 0

        def clean(raw: str) -> Optional[str]:
            line = self.normalize_text(raw.rstrip("\r\n"), config)
            if not line:
                return None
            return line

        if config.shuffle:
            with open(path, "r", encoding=config.encoding) as f:
                lines = [line for line in (clean(raw) for raw in f) if line is not None]
            rng = random.Random(int(config.seed))
            rng.shuffle(lines)
            for line in lines:
                if max_chars is not None and emitted >= max_chars:
                    break
                if max_chars is not None and emitted + len(line) > max_chars:
                    line = line[: max(0, max_chars - emitted)]
                if line:
                    emitted += len(line)
                    yield line
            return

        while True:
            any_line = False
            with open(path, "r", encoding=config.encoding) as f:
                for raw in f:
                    line = clean(raw)
                    if line is None:
                        continue
                    any_line = True
                    if max_chars is not None and emitted >= max_chars:
                        return
                    if max_chars is not None and emitted + len(line) > max_chars:
                        line = line[: max(0, max_chars - emitted)]
                    if line:
                        emitted += len(line)
                        yield line
            if not config.repeat or not any_line:
                return

    def load_text(self, config: DatasetConfig, separator: str = "\n") -> str:
        return separator.join(self.iter_lines(config))

    def load_text_splits(
        self,
        name: str,
        train_fraction: float = 0.70,
        max_train_chars: Optional[int] = None,
        max_eval_chars: Optional[int] = None,
        seed: int = 42,
    ) -> DatasetTextSplits:
        source, dataset, _ = self._source_and_split(name, "train")
        if source is None:
            path = str(self.resolve_path(name))
            text = self.load_text(DatasetConfig(name=name, max_chars=_sum_optional(max_train_chars, 2 * (max_eval_chars or 0))))
            train, val, test = _split_text(text, train_fraction)
            return DatasetTextSplits(dataset, train, val, test, path, path, path, "unknown")

        train_path = self.resolve_path(dataset, "train")
        val_path = self.resolve_path(dataset, "validation")
        test_path = self.resolve_path(dataset, "test")
        same_file = train_path == val_path == test_path
        if same_file:
            max_chars = _sum_optional(max_train_chars, 2 * (max_eval_chars or 0))
            text = self.load_text(DatasetConfig(name=dataset, split="train", max_chars=max_chars, seed=seed))
            train, val, test = _split_text(text, train_fraction)
        else:
            train = self.load_text(DatasetConfig(name=dataset, split="train", max_chars=max_train_chars, seed=seed))
            val = self.load_text(DatasetConfig(name=dataset, split="validation", max_chars=max_eval_chars, seed=seed))
            test = self.load_text(DatasetConfig(name=dataset, split="test", max_chars=max_eval_chars, seed=seed))
        return DatasetTextSplits(
            dataset=dataset,
            train=train,
            validation=val,
            test=test,
            train_path=str(train_path),
            validation_path=str(val_path),
            test_path=str(test_path),
            language=source.language,
        )

    def batch_lines(self, config: DatasetConfig, batch_size: int = 1000) -> Iterator[List[str]]:
        batch: List[str] = []
        for line in self.iter_lines(config):
            batch.append(line)
            if len(batch) >= int(batch_size):
                yield batch
                batch = []
        if batch:
            yield batch

    def token_batches(
        self,
        config: DatasetConfig,
        tokenizer,
        batch_tokens: int = 2048,
        max_tokens: Optional[int] = None,
    ) -> Iterator[List[int]]:
        batch: List[int] = []
        emitted = 0
        limit = None if max_tokens is None else int(max_tokens)
        for line in self.iter_lines(config):
            remaining = None if limit is None else max(0, limit - emitted)
            if remaining == 0:
                break
            try:
                tokens = tokenizer.encode(line, max_tokens=remaining)
            except TypeError:
                tokens = tokenizer.encode(line)
                if remaining is not None:
                    tokens = tokens[:remaining]
            for token in tokens:
                batch.append(int(token))
                emitted += 1
                if len(batch) >= int(batch_tokens):
                    yield batch
                    batch = []
                if limit is not None and emitted >= limit:
                    break
            if limit is not None and emitted >= limit:
                break
        if batch:
            yield batch

    def load_tokens(self, config: DatasetConfig, tokenizer, max_tokens: Optional[int] = None) -> List[int]:
        out: List[int] = []
        for batch in self.token_batches(config, tokenizer, batch_tokens=8192, max_tokens=max_tokens):
            out.extend(batch)
        return out

    def compute_stats(self, config: DatasetConfig) -> DatasetStats:
        path = self.resolve_path(config.name, config.split)
        source, split = self._config_for(config)
        stats = DatasetStats(
            name=source.name,
            split=split,
            path=str(path),
            language=config.language or source.language,
        )
        min_len: Optional[int] = None
        for line in self.iter_lines(DatasetConfig(**{**config.__dict__, "max_chars": None, "shuffle": False, "repeat": False})):
            length = len(line)
            stats.total_lines += 1
            stats.total_chars += length
            stats.total_words += len(line.split())
            min_len = length if min_len is None else min(min_len, length)
            stats.max_line_length = max(stats.max_line_length, length)
        stats.min_line_length = int(min_len or 0)
        stats.avg_line_length = stats.total_chars / max(1, stats.total_lines)
        self.stats[(source.name, split)] = stats
        return stats

    def list_available_datasets(self) -> List[str]:
        available: List[str] = []
        for name in sorted(self.DATASETS):
            try:
                self.resolve_path(name, "train")
                available.append(name)
            except FileNotFoundError:
                continue
        return available

    def get_all_stats(self) -> Dict[str, Dict[str, object]]:
        stats: Dict[str, Dict[str, object]] = {}
        for name in self.list_available_datasets():
            try:
                stats[name] = self.compute_stats(DatasetConfig(name=name)).to_dict()
            except Exception as exc:  # pragma: no cover - diagnostics path
                stats[name] = {"error": str(exc)}
        return stats


def _sum_optional(left: Optional[int], right: int) -> Optional[int]:
    if left is None and right <= 0:
        return None
    return int(left or 0) + int(right)


def _split_text(text: str, train_fraction: float) -> Tuple[str, str, str]:
    if not text:
        return "", "", ""
    train_fraction = min(0.95, max(0.05, float(train_fraction)))
    train_cut = max(1, min(len(text) - 1, int(len(text) * train_fraction)))
    rest = text[train_cut:]
    val_cut = max(1, len(rest) // 2) if rest else 0
    return text[:train_cut], rest[:val_cut], rest[val_cut:]


def create_loader(root_dir: Optional[str] = None) -> DatasetLoader:
    return DatasetLoader(root_dir=root_dir)


def load_tinystories(max_chars: Optional[int] = None, shuffle: bool = False, seed: int = 42) -> Iterator[str]:
    loader = create_loader()
    return loader.iter_lines(DatasetConfig("tinystories", max_chars=max_chars, shuffle=shuffle, seed=seed))


def load_wikitext2(
    split: str = "train",
    max_chars: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterator[str]:
    loader = create_loader()
    return loader.iter_lines(DatasetConfig("wikitext2", split=split, max_chars=max_chars, shuffle=shuffle, seed=seed))
