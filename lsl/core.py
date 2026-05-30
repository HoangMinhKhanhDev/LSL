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
from . import sparse_native
from .sparse_native import NATIVE_AVAILABLE
from .synapse import LivingSynapseLayer


@dataclass
class LSLCoreConfig:
    vocab_size: int = 8000
    tokenizer: str = "subword"
    candidate_cap: int = 128
    seed: int = 42
    runtime_profile: str = "full"
    consolidation_interval: int = 4096
    consolidation_fraction: float = 0.02
    use_native_core: bool = True
    native_max_vocab: int = 4096
    native_lr: float = 0.35
    native_decay: float = 0.0005
    native_score_weight: float = 2.0


class LSLCoreModel:
    """One public model surface for training, generation, QA, and diagnostics."""

    def __init__(self, config: Optional[LSLCoreConfig] = None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword settings, not both")
        self.config = config or LSLCoreConfig(**kwargs)
        full_modules = self._runtime_profile_from_config(self.config) == "full"
        self.agent = BioComputeAgent(
            vocab_size=self.config.vocab_size,
            tokenizer=self.config.tokenizer,
            candidate_cap=self.config.candidate_cap,
            seed=self.config.seed,
            use_pc_v2=full_modules,
            use_sdr_v2=full_modules,
            use_columns=full_modules,
            use_neuromodulation=full_modules,
        )
        self.tokenizer_built = False
        self.prev_token: Optional[int] = None
        self.seen_tokens = 0
        self.training_seconds = 0.0
        self.last_metrics: Dict[str, float] = {}
        self.native_transition: Optional[LivingSynapseLayer] = None
        self.native_core_reason = "not initialized"
        self.native_core_stats: Dict[str, float] = {
            "forward_calls": 0.0,
            "native_forward_calls": 0.0,
            "forward_touched": 0.0,
            "update_calls": 0.0,
            "native_update_calls": 0.0,
            "update_touched": 0.0,
        }

    def _upgrade_runtime(self) -> None:
        defaults = LSLCoreConfig()
        for key, value in defaults.__dict__.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)
        if not hasattr(self, "native_transition"):
            self.native_transition = None
        if not hasattr(self, "native_core_reason"):
            self.native_core_reason = "not initialized"
        if not hasattr(self, "native_core_stats"):
            self.native_core_stats = {}
        for key, value in {
            "forward_calls": 0.0,
            "native_forward_calls": 0.0,
            "forward_touched": 0.0,
            "update_calls": 0.0,
            "native_update_calls": 0.0,
            "update_touched": 0.0,
        }.items():
            self.native_core_stats.setdefault(key, value)

    @staticmethod
    def _runtime_profile_from_config(config: LSLCoreConfig) -> str:
        profile = str(getattr(config, "runtime_profile", "full")).strip().lower().replace("-", "_")
        if profile not in {"full", "native_long_context", "native_fast"}:
            raise ValueError(f"Unsupported LSL runtime profile: {profile}")
        return profile

    def runtime_profile(self) -> str:
        self._upgrade_runtime()
        return self._runtime_profile_from_config(self.config)

    def set_runtime_profile(self, profile: str) -> None:
        self.config.runtime_profile = str(profile).strip().lower().replace("-", "_")
        full_modules = self.runtime_profile() == "full"
        self.agent.use_pc_v2 = full_modules
        self.agent.use_sdr_v2 = full_modules
        self.agent.use_columns = full_modules
        self.agent.use_neuromodulation = full_modules
        if not full_modules:
            self.config.consolidation_interval = 0

    @property
    def vocab_size(self) -> int:
        return int(self.agent.vocab_size)

    @property
    def tokenizer(self):
        return self.agent.tokenizer

    def build_tokenizer(self, text: str) -> None:
        self.agent.build_tokenizer(text)
        self.tokenizer_built = True
        self._ensure_native_core()

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
        if self.native_transition is not None:
            self.native_transition.fatigue.fill(0.0)

    def _ensure_native_core(self) -> bool:
        self._upgrade_runtime()
        if not self.config.use_native_core:
            self.native_transition = None
            self.native_core_reason = "disabled by config"
            return False
        if not NATIVE_AVAILABLE:
            self.native_transition = None
            self.native_core_reason = "lsl._sparse_native unavailable"
            return False
        vocab = int(self.vocab_size)
        if vocab <= 1:
            self.native_transition = None
            self.native_core_reason = "vocab too small"
            return False
        if vocab > int(self.config.native_max_vocab):
            self.native_transition = None
            self.native_core_reason = f"vocab {vocab} exceeds native_max_vocab {self.config.native_max_vocab}"
            return False
        if (
            self.native_transition is not None
            and self.native_transition.in_dim == vocab
            and self.native_transition.out_dim == vocab
        ):
            self.native_core_reason = "enabled"
            return True
        self.native_transition = LivingSynapseLayer(
            vocab,
            vocab,
            slow_init=0.0,
            seed=int(self.config.seed) + 1009,
        )
        self.native_core_reason = "enabled"
        return True

    def _record_native_forward(self, stats: Dict[str, int]) -> None:
        self.native_core_stats["forward_calls"] += 1.0
        self.native_core_stats["forward_touched"] += float(stats.get("fatigue_touched", stats.get("touched", 0)))
        if str(stats.get("mode", "")).startswith("native_"):
            self.native_core_stats["native_forward_calls"] += 1.0

    def _record_native_update(self, stats: Dict[str, int]) -> None:
        self.native_core_stats["update_calls"] += 1.0
        self.native_core_stats["update_touched"] += float(stats.get("weights_touched", stats.get("touched", 0)))
        if str(stats.get("mode", "")).startswith("native_"):
            self.native_core_stats["native_update_calls"] += 1.0

    def _native_scores(self, token_id: int) -> Optional[np.ndarray]:
        if not self._ensure_native_core() or self.native_transition is None:
            return None
        post, stats = self.native_transition.forward_active(
            np.array([int(token_id) % self.vocab_size], dtype=np.intp),
            np.ones(1, dtype=np.float32),
            return_stats=True,
        )
        self._record_native_forward(stats)
        return post

    def _native_observe_transition(self, source: int, target: int) -> None:
        if not self._ensure_native_core() or self.native_transition is None:
            return
        self.native_transition.target_update_from_active(
            np.array([int(source) % self.vocab_size], dtype=np.intp),
            int(target) % self.vocab_size,
            np.ones(1, dtype=np.float32),
            lr=float(self.config.native_lr),
            decay=float(self.config.native_decay),
            max_abs=12.0,
        )
        self._record_native_update(self.native_transition.last_update_ops)

    def _native_predict(self, token_id: int) -> tuple[Optional[int], float]:
        summary = self._native_score_summary(token_id)
        if summary is None:
            return None, 0.0
        best = int(summary.get("best_index", -1))
        value = float(summary.get("best_score", 0.0))
        if value <= 1.0e-8:
            return None, 0.0
        return best, value

    def _native_score_summary(self, token_id: int, target_id: int = -1) -> Optional[Dict[str, float]]:
        if not self._ensure_native_core() or self.native_transition is None:
            return None
        try:
            stats = sparse_native.score_active(
                self.native_transition.W_slow,
                self.native_transition.W_live,
                self.native_transition.fatigue,
                np.array([int(token_id) % self.vocab_size], dtype=np.intp),
                np.ones(1, dtype=np.float32),
                int(target_id),
            )
        except RuntimeError:
            scores = self._native_scores(token_id)
            if scores is None:
                return None
            positive = np.maximum(scores.astype(np.float32, copy=False), 0.0)
            best = int(np.argmax(scores))
            target = int(target_id)
            return {
                "mode": "python_sparse_active_score",
                "best_index": float(best),
                "best_score": float(scores[best]),
                "target_score": float(positive[target]) if 0 <= target < len(positive) else 0.0,
                "positive_sum": float(np.sum(positive)),
                "touched": float(len(scores)),
            }
        self._record_native_forward({
            "mode": stats.get("mode", "native_sparse_active_score"),
            "fatigue_touched": int(stats.get("touched", self.vocab_size)),
        })
        return stats

    def rebuild_native_core_from_memory(self) -> Dict[str, float]:
        """Pack learned one-token transitions into the native C sparse head.

        This upgrades older checkpoints that learned Python sparse transition
        counts before the native chat path existed.
        """
        if not self._ensure_native_core() or self.native_transition is None:
            return {"rebuilt_sources": 0.0, "rebuilt_edges": 0.0, "native_enabled": 0.0}
        memory = getattr(self.agent, "long_context", None)
        counts = getattr(memory, "_unigram_counts", None)
        if not counts:
            return {"rebuilt_sources": 0.0, "rebuilt_edges": 0.0, "native_enabled": 1.0}
        self.native_transition.W_live.fill(0.0)
        self.native_transition.fatigue.fill(0.0)
        rebuilt_sources = 0
        rebuilt_edges = 0
        vocab = int(self.vocab_size)
        for source in range(vocab):
            key = memory._unigram_key(source)
            bucket = counts.get(key)
            if not bucket:
                continue
            total = float(sum(bucket.values()))
            if total <= 0.0:
                continue
            for target, count in bucket.items():
                target = int(target)
                if 0 <= target < vocab:
                    self.native_transition.target_update_from_active(
                        np.array([source], dtype=np.intp),
                        target,
                        np.ones(1, dtype=np.float32),
                        lr=float(count) / total,
                        decay=0.0,
                        max_abs=12.0,
                    )
                    self._record_native_update(self.native_transition.last_update_ops)
                    rebuilt_edges += 1
            rebuilt_sources += 1
        return {
            "rebuilt_sources": float(rebuilt_sources),
            "rebuilt_edges": float(rebuilt_edges),
            "native_enabled": 1.0,
        }

    def observe_token(self, token_id: int, learn: bool = True) -> None:
        profile = self.runtime_profile()
        token = int(token_id) % max(1, self.vocab_size)
        if learn and self.prev_token is not None:
            self._native_observe_transition(self.prev_token, token)
            if profile != "native_fast":
                self.agent.long_context.observe_transition(self.prev_token, token, vocab_size=self.vocab_size)
                self.agent.homeostasis.observe(active_count=1, total_count=max(1, self.vocab_size), local_error=0.10)

        if profile == "full" and self.agent.use_sdr_v2:
            bits = self.agent.sdr_v2.encode(str(token))
            self.agent.sdr_observations += 1
            self.agent.sdr_active_total += len(bits)

        if profile == "full" and self.agent.use_columns and token < self.agent.columns.vocab_size:
            self.agent.columns.forward(token, learn=learn)

        if profile == "full" and self.agent.use_pc_v2:
            states = [
                self.agent.pc_v2.state_for(token + layer * 997, layer)
                for layer in range(self.agent.pc_v2.layers)
            ]
            self.agent.pc_v2.observe(states, learn=learn)

        if profile == "full" and learn and self.agent.use_neuromodulation:
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
        if self.runtime_profile() != "native_fast":
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
            if self.runtime_profile() != "native_fast":
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
        if not tokens:
            return None
        if self.runtime_profile() == "native_fast":
            native_token, _ = self._native_predict(tokens[-1])
            return native_token
        votes: Dict[int, float] = {}
        native_token, native_score = self._native_predict(tokens[-1])
        if native_token is not None:
            votes[int(native_token)] = votes.get(int(native_token), 0.0) + float(native_score) * float(self.config.native_score_weight)
        agent_token = self.agent.predict_next_token_id(tokens)
        if agent_token is not None:
            votes[int(agent_token)] = votes.get(int(agent_token), 0.0) + 1.0
        if not votes:
            return None
        return int(max(votes.items(), key=lambda item: (item[1], -item[0]))[0])

    def _native_probability_and_prediction(self, current: int, target: int) -> tuple[float, Optional[int]]:
        vocab = max(1, int(self.vocab_size))
        summary = self._native_score_summary(int(current), int(target))
        if summary is None:
            return 1.0 / vocab, None
        total = float(summary.get("positive_sum", 0.0))
        if total <= 1.0e-8:
            return 1.0 / vocab, None
        alpha = 0.05
        prob = (float(summary.get("target_score", 0.0)) + alpha) / (total + alpha * vocab)
        return float(max(prob, 1e-12)), int(summary.get("best_index", -1))

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
            if self.runtime_profile() == "native_fast":
                prob, pred = self._native_probability_and_prediction(current, target)
            else:
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
        tokens = self.encode(prompt)
        if not tokens:
            generated = self.agent.generate(prompt, max_new_tokens=max_new_tokens)
            return generated
        out = [int(token) for token in tokens]
        trigrams = {tuple(out[i:i + 3]) for i in range(max(0, len(out) - 2))}
        for _ in range(max(0, int(max_new_tokens))):
            nxt = self.predict_next_token_id(out)
            if nxt is None:
                break
            tri = tuple((out + [int(nxt)])[-3:])
            if stop_on_trigram_loop and len(tri) == 3 and tri in trigrams:
                break
            out.append(int(nxt))
            if len(out) >= 3:
                trigrams.add(tuple(out[-3:]))
        if len(out) == len(tokens):
            generated = self.agent.generate(prompt, max_new_tokens=max_new_tokens)
            if not stop_on_trigram_loop:
                return generated
            out = self.encode(generated)
        trimmed: List[int] = []
        trigrams = set()
        for token in out:
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
        self._upgrade_runtime()
        forward_calls = self.native_core_stats.get("forward_calls", 0.0)
        update_calls = self.native_core_stats.get("update_calls", 0.0)
        return {
            "seen_tokens": float(self.seen_tokens),
            "training_seconds": float(self.training_seconds),
            "runtime_profile": self.runtime_profile(),
            "native_core_available": float(NATIVE_AVAILABLE),
            "native_core_enabled": float(self.native_transition is not None),
            "native_core_vocab": float(self.native_transition.in_dim if self.native_transition is not None else 0),
            "native_core_forward_calls": float(forward_calls),
            "native_core_update_calls": float(update_calls),
            "native_core_forward_native_ratio": self.native_core_stats.get("native_forward_calls", 0.0) / max(1.0, forward_calls),
            "native_core_update_native_ratio": self.native_core_stats.get("native_update_calls", 0.0) / max(1.0, update_calls),
            "native_core_forward_touched": float(self.native_core_stats.get("forward_touched", 0.0)),
            "native_core_update_touched": float(self.native_core_stats.get("update_touched", 0.0)),
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
        model._upgrade_runtime()
        if getattr(model, "tokenizer_built", False):
            model._ensure_native_core()
        return model
