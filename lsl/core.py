"""Unified LSL core model.

This module turns the currently separate biological mechanisms into one
model-facing API. The implementation keeps the strict path local and online:
tokens enter once, every module sees the same token stream, and updates happen
at active local state only.
"""
from __future__ import annotations

import math
import os
import pickle
import time
import base64
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .bio import BioComputeAgent
from .generation import GenerationController


@dataclass
class LSLCoreConfig:
    vocab_size: int = 8000
    tokenizer: str = "subword"
    candidate_cap: int = 128
    seed: int = 42
    consolidation_interval: int = 4096
    consolidation_fraction: float = 0.02


class LSLCoreModel:
    """One public model surface for training, generation, QA, and diagnostics."""

    def __init__(self, config: Optional[LSLCoreConfig] = None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword settings, not both")
        self.config = config or LSLCoreConfig(**kwargs)
        self.agent = BioComputeAgent(
            vocab_size=self.config.vocab_size,
            tokenizer=self.config.tokenizer,
            candidate_cap=self.config.candidate_cap,
            seed=self.config.seed,
        )
        self.tokenizer_built = False
        self.prev_token: Optional[int] = None
        self.seen_tokens = 0
        self.training_seconds = 0.0
        self.last_metrics: Dict[str, float] = {}

    @property
    def vocab_size(self) -> int:
        return int(self.agent.vocab_size)

    @property
    def tokenizer(self):
        return self.agent.tokenizer

    def build_tokenizer(self, text: str) -> None:
        self.agent.build_tokenizer(text)
        self.tokenizer_built = True

    def encode(self, text: str) -> List[int]:
        if not self.tokenizer_built:
            self.build_tokenizer(text)
        return [int(token) for token in self.agent.tokenizer.encode(text)]

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.agent.tokenizer.decode([int(token) for token in token_ids])

    def reset_state(self) -> None:
        self.prev_token = None
        self.agent.long_context.reset_state()
        if self.agent.use_columns:
            self.agent.columns.reset_state()
        if self.agent.use_pc_v2:
            self.agent.pc_v2.reset_state()

    def observe_token(self, token_id: int, learn: bool = True) -> None:
        token = int(token_id) % max(1, self.vocab_size)
        if learn and self.prev_token is not None:
            self.agent.long_context.observe_transition(self.prev_token, token, vocab_size=self.vocab_size)
            self.agent.homeostasis.observe(active_count=1, total_count=max(1, self.vocab_size), local_error=0.10)

        if self.agent.use_sdr_v2:
            bits = self.agent.sdr_v2.encode(str(token))
            self.agent.sdr_observations += 1
            self.agent.sdr_active_total += len(bits)

        if self.agent.use_columns and token < self.agent.columns.vocab_size:
            self.agent.columns.forward(token, learn=learn)

        if self.agent.use_pc_v2:
            states = [
                self.agent.pc_v2.state_for(token + layer * 997, layer)
                for layer in range(self.agent.pc_v2.layers)
            ]
            self.agent.pc_v2.observe(states, learn=learn)

        if learn and self.agent.use_neuromodulation:
            surprise = 0.8 if self.agent.bio_modulator.seen[str(token)] == 0 else 0.05
            self.agent.bio_modulator.observe(str(token), surprise=surprise)

        self.prev_token = token
        self.seen_tokens += 1
        self.agent.generator = None
        if (
            learn
            and self.config.consolidation_interval > 0
            and self.seen_tokens % int(self.config.consolidation_interval) == 0
        ):
            self.agent.consolidate(replay_fraction=self.config.consolidation_fraction)

    def fit_tokens(self, tokens: Sequence[int], reset: bool = True) -> Dict[str, float]:
        if reset:
            self.reset_state()
        t0 = time.perf_counter()
        for token in tokens:
            self.observe_token(int(token), learn=True)
        elapsed = time.perf_counter() - t0
        self.training_seconds += elapsed
        metrics = {
            "tokens": float(len(tokens)),
            "elapsed_seconds": float(elapsed),
            "us_per_token": 1_000_000.0 * elapsed / max(1, len(tokens)),
        }
        self.last_metrics = metrics
        return metrics

    def observe(self, text: str, source: str = "core", learn: bool = True) -> Dict[str, float]:
        if not self.tokenizer_built:
            self.build_tokenizer(text)
        self.agent.world.observe_chunk(text, source=source)
        tokens = self.encode(text)
        if not learn:
            return {"tokens": float(len(tokens)), "elapsed_seconds": 0.0, "us_per_token": 0.0}
        return self.fit_tokens(tokens, reset=False)

    def train_stream(
        self,
        texts: Iterable[str],
        tokenizer_text_chars: int = 250000,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, float]:
        items = [str(text) for text in texts]
        if not items:
            return {"tokens": 0.0, "elapsed_seconds": 0.0, "us_per_token": 0.0}
        if not self.tokenizer_built:
            seed_text = " ".join(items)[: int(tokenizer_text_chars)]
            self.build_tokenizer(seed_text)

        total_tokens = 0
        total_seconds = 0.0
        self.reset_state()
        for idx, text in enumerate(items):
            self.agent.world.observe_chunk(text, source=f"stream:{idx}")
            tokens = self.encode(text)
            if max_tokens is not None:
                remaining = int(max_tokens) - total_tokens
                if remaining <= 0:
                    break
                tokens = tokens[:remaining]
            metrics = self.fit_tokens(tokens, reset=False)
            total_tokens += len(tokens)
            total_seconds += metrics["elapsed_seconds"]
            if max_tokens is not None and total_tokens >= int(max_tokens):
                break
        out = {
            "tokens": float(total_tokens),
            "elapsed_seconds": float(total_seconds),
            "us_per_token": 1_000_000.0 * total_seconds / max(1, total_tokens),
        }
        self.last_metrics = out
        return out

    def predict_next_token_id(self, prompt: Sequence[int] | str) -> Optional[int]:
        tokens = self.encode(prompt) if isinstance(prompt, str) else [int(x) for x in prompt]
        return self.agent.predict_next_token_id(tokens)

    def evaluate_tokens(self, tokens: Sequence[int], update_context: bool = True) -> Dict[str, float]:
        items = [int(token) % max(1, self.vocab_size) for token in tokens]
        if len(items) < 2:
            return {"loss": float("inf"), "perplexity": float("inf"), "accuracy": 0.0, "p50_latency_us": 0.0, "tokens": float(len(items))}
        self.agent.long_context.reset_state()
        losses: List[float] = []
        correct = 0
        times: List[float] = []
        for current, target in zip(items, items[1:]):
            t0 = time.perf_counter_ns()
            prob = self.agent.long_context.target_probability(
                current,
                target,
                vocab_size=self.vocab_size,
                update_context=update_context,
            )
            pred = self.agent.long_context.top_next(current, vocab_size=self.vocab_size)
            times.append((time.perf_counter_ns() - t0) / 1000.0)
            losses.append(-math.log(max(float(prob), 1e-12)))
            correct += int(pred == target)
        loss = float(np.mean(losses)) if losses else float("inf")
        return {
            "loss": loss,
            "perplexity": float(math.exp(min(20.0, loss))),
            "accuracy": float(correct / max(1, len(losses))),
            "p50_latency_us": float(np.percentile(times, 50)) if times else 0.0,
            "tokens": float(len(items)),
        }

    def evaluate_text(self, text: str, max_tokens: Optional[int] = None) -> Dict[str, float]:
        tokens = self.encode(text)
        if max_tokens is not None:
            tokens = tokens[: int(max_tokens)]
        metrics = self.evaluate_tokens(tokens)
        unk_id = getattr(self.tokenizer, "word_to_id", getattr(self.tokenizer, "token_to_id", {})).get("<UNK>", 1)
        metrics["unk_rate"] = sum(int(token) == int(unk_id) for token in tokens) / max(1, len(tokens))
        return metrics

    def generate(self, prompt: str, max_new_tokens: int = 64, stop_on_trigram_loop: bool = True) -> str:
        generated = self.agent.generate(prompt, max_new_tokens=max_new_tokens)
        if not stop_on_trigram_loop:
            return generated
        tokens = self.encode(generated)
        trimmed: List[int] = []
        trigrams = set()
        for token in tokens:
            trimmed.append(int(token))
            if len(trimmed) < 3:
                continue
            tri = tuple(trimmed[-3:])
            if tri in trigrams:
                trimmed = trimmed[:-1]
                break
            trigrams.add(tri)
        return self.decode(trimmed)

    def generation_metrics(self, text: str) -> Dict[str, float]:
        tokens = self.encode(text)
        unk_id = getattr(self.tokenizer, "word_to_id", getattr(self.tokenizer, "token_to_id", {})).get("<UNK>", 1)
        return GenerationController.generation_metrics(tokens, unk_id=unk_id)

    def answer(self, question: str):
        return self.agent.answer(question)

    def diagnostics(self) -> Dict[str, float]:
        return {
            "seen_tokens": float(self.seen_tokens),
            "training_seconds": float(self.training_seconds),
            **self.agent.diagnostics(),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        if path.lower().endswith(".json"):
            wrapped = {
                "format": "LSLCoreModelBase64Pickle",
                "version": 1,
                "vocab_size": self.vocab_size,
                "seen_tokens": self.seen_tokens,
                "training_seconds": self.training_seconds,
                "payload_b64": base64.b64encode(payload).decode("ascii"),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(wrapped, f)
            return
        with open(path, "wb") as f:
            f.write(payload)

    @classmethod
    def load(cls, path: str) -> "LSLCoreModel":
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                wrapped = json.load(f)
            if wrapped.get("format") != "LSLCoreModelBase64Pickle":
                raise ValueError("Unsupported LSLCoreModel JSON checkpoint format")
            raw = base64.b64decode(wrapped["payload_b64"])
            model = pickle.loads(raw)
        else:
            with open(path, "rb") as f:
                model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected LSLCoreModel checkpoint, got {type(model)!r}")
        return model
