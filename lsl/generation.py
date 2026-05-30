"""Open text generation control built on local sparse memories."""
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .long_context import LongContextMemory


@dataclass
class DiscourseState:
    """Rolling local state used to score generation candidates."""

    recent_tokens: deque = field(default_factory=lambda: deque(maxlen=96))
    token_fatigue: Counter = field(default_factory=Counter)
    bigrams: Counter = field(default_factory=Counter)
    trigrams: Counter = field(default_factory=Counter)
    topic_tokens: Counter = field(default_factory=Counter)

    def observe(self, token_id: int) -> None:
        token_id = int(token_id)
        if len(self.recent_tokens) >= 1:
            self.bigrams[(self.recent_tokens[-1], token_id)] += 1
        if len(self.recent_tokens) >= 2:
            self.trigrams[(self.recent_tokens[-2], self.recent_tokens[-1], token_id)] += 1
        self.recent_tokens.append(token_id)
        self.token_fatigue[token_id] += 1
        self.topic_tokens[token_id] += 1


@dataclass
class DiscoursePlan:
    target_length: int = 64
    topic_window: int = 64
    entity_ids: Tuple[int, ...] = ()
    contradiction_pairs: Tuple[Tuple[int, int], ...] = ()
    style_tokens: Tuple[int, ...] = ()


class GenerationController:
    """Local candidate scorer for open generation.

    The controller never builds a pairwise context matrix. It asks the
    long-context memory for a bounded candidate set, then scores those
    candidates with local transition probability, discourse continuity, and
    repetition fatigue.
    """

    def __init__(
        self,
        memory: Optional[LongContextMemory] = None,
        vocab_size: int = 1000,
        candidate_limit: int = 16,
        unk_id: int = 1,
        sentence_end_ids: Optional[Iterable[int]] = None,
        plan: Optional[DiscoursePlan] = None,
        seed: int = 0,
    ):
        self.vocab_size = int(vocab_size)
        self.memory = memory or LongContextMemory(vocab_size=self.vocab_size, seed=seed)
        self.candidate_limit = max(2, int(candidate_limit))
        self.unk_id = int(unk_id)
        self.sentence_end_ids = {int(x) for x in sentence_end_ids or []}
        self.plan = plan or DiscoursePlan()
        self.rng = np.random.default_rng(seed)
        self.last_scores: Dict[int, float] = {}

    def observe_sequence(self, tokens: Sequence[int]) -> None:
        items = [int(t) for t in tokens]
        for i in range(len(items) - 1):
            self.memory.observe_transition(items[i], items[i + 1], vocab_size=self.vocab_size)

    def _prepare_state(self, prompt: Sequence[int]) -> DiscourseState:
        state = DiscourseState()
        for token in prompt:
            state.observe(int(token))
        return state

    def _prime_memory_context(self, prompt: Sequence[int]) -> None:
        self.memory.reset_state()
        for token in list(prompt)[:-1]:
            self.memory.advance_context(int(token))

    def candidate_scores(self, current: int, state: DiscourseState) -> Dict[int, float]:
        candidates = list(self.memory.next_candidates(int(current), limit=self.candidate_limit))
        if not candidates:
            fallback = self.memory.top_next(int(current), vocab_size=self.vocab_size)
            candidates = [] if fallback is None else [int(fallback)]
        if not candidates:
            return {}

        recent = list(state.recent_tokens)
        topic_total = max(1, sum(state.topic_tokens.values()))
        scores: Dict[int, float] = {}
        for rank, candidate in enumerate(candidates):
            candidate = int(candidate)
            prob = self.memory.target_probability(
                int(current),
                candidate,
                vocab_size=self.vocab_size,
                update_context=False,
            )
            score = 2.0 * np.log(max(float(prob), 1e-12))
            score += 1.0 / float(rank + 1)
            if candidate != self.unk_id:
                score += 0.25
            if candidate in self.sentence_end_ids and len(recent) >= min(10, self.plan.target_length):
                score += 0.08
            if state.topic_tokens.get(candidate, 0):
                score += 0.08 * float(state.topic_tokens[candidate]) / topic_total
            if candidate in self.plan.entity_ids:
                score += 0.15
            if candidate in self.plan.style_tokens:
                score += 0.06
            for a, b in self.plan.contradiction_pairs:
                if candidate == int(a) and state.token_fatigue.get(int(b), 0):
                    score -= 1.2
                if candidate == int(b) and state.token_fatigue.get(int(a), 0):
                    score -= 1.2

            fatigue = state.token_fatigue.get(candidate, 0)
            score -= 0.20 * min(6, fatigue)
            if len(recent) >= 1 and state.bigrams.get((recent[-1], candidate), 0):
                score -= 0.60 * state.bigrams[(recent[-1], candidate)]
            if len(recent) >= 2 and state.trigrams.get((recent[-2], recent[-1], candidate), 0):
                score -= 2.50 * state.trigrams[(recent[-2], recent[-1], candidate)]
            scores[candidate] = float(score)
        self.last_scores = scores
        return scores

    def choose_next(self, current: int, state: DiscourseState) -> Optional[int]:
        scores = self.candidate_scores(int(current), state)
        if not scores:
            return None
        return int(max(scores.items(), key=lambda item: (item[1], -item[0]))[0])

    def generate(self, prompt: Sequence[int], max_new_tokens: int = 48, plan: Optional[DiscoursePlan] = None) -> List[int]:
        if not prompt:
            return []
        old_plan = self.plan
        if plan is not None:
            self.plan = plan
        out = [int(t) for t in prompt]
        state = self._prepare_state(out)
        self._prime_memory_context(out)
        current = int(out[-1])
        for _ in range(max(0, int(max_new_tokens))):
            nxt = self.choose_next(current, state)
            if nxt is None:
                break
            self.memory.advance_context(current)
            out.append(int(nxt))
            state.observe(int(nxt))
            current = int(nxt)
        self.plan = old_plan
        return out

    @staticmethod
    def generation_metrics(
        tokens: Sequence[int],
        unk_id: int = 1,
        entity_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, float]:
        items = [int(t) for t in tokens]
        if len(items) < 4:
            return {
                "length": float(len(items)),
                "unk_rate": 1.0,
                "loop_rate": 1.0,
                "distinct2": 0.0,
                "coherence": 0.0,
            }
        bigrams = list(zip(items, items[1:]))
        trigrams = [tuple(items[i:i + 3]) for i in range(len(items) - 2)]
        unk_rate = sum(1 for token in items if token == int(unk_id)) / max(1, len(items))
        loop_rate = 1.0 - len(set(trigrams)) / max(1, len(trigrams))
        distinct2 = len(set(bigrams)) / max(1, len(bigrams))
        length_score = min(1.0, len(items) / 48.0)
        entity_set = {int(x) for x in entity_ids or []}
        if entity_set:
            entity_hits = sum(1 for token in items if token in entity_set)
            entity_consistency = min(1.0, entity_hits / max(1, len(entity_set)))
        else:
            entity_consistency = 1.0
        coherence = (
            0.30 * distinct2
            + 0.30 * (1.0 - loop_rate)
            + 0.20 * (1.0 - unk_rate)
            + 0.10 * length_score
            + 0.10 * entity_consistency
        )
        return {
            "length": float(len(items)),
            "unk_rate": float(unk_rate),
            "loop_rate": float(loop_rate),
            "distinct2": float(distinct2),
            "entity_consistency": float(entity_consistency),
            "coherence": float(coherence),
        }
