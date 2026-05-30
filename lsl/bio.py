"""Phase 9 biological compute primitives.

These components are strict-path helpers: local state, online updates, bounded
candidate lookup, and sparse active-index computation only.
"""
import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .agent import IntegratedLSLAgent
from .cortical_column import CorticalColumnSequenceMemory


def _norm(value) -> str:
    return str(value).strip().lower()


def _hash_u64(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little")


def _feature_bits(feature: str, dim: int, count: int, seed: int = 0) -> Tuple[int, ...]:
    bits = set()
    nonce = 0
    while len(bits) < int(count):
        bits.add(int(_hash_u64(seed, feature, nonce) % int(dim)))
        nonce += 1
    return tuple(sorted(bits))


class LocalPredictiveStack:
    """Layer-local predictor with exact local pre-state -> post-state tables."""

    def __init__(self, layers: int = 3, width: int = 256, k: int = 8, theta: float = 0.05):
        self.layers = int(layers)
        self.width = int(width)
        self.k = int(k)
        self.theta = float(theta)
        self.tables: List[Dict[Tuple[int, ...], Counter]] = [defaultdict(Counter) for _ in range(self.layers)]
        self.prev_states: List[Optional[Tuple[int, ...]]] = [None for _ in range(self.layers)]
        self.error_history: List[List[float]] = [[] for _ in range(self.layers)]
        self.update_count = 0
        self.step_count = 0
        self.zero_update_count = 0

    def state_for(self, token: int, layer: int = 0) -> Tuple[int, ...]:
        base = int(token)
        return _feature_bits(f"pc:{layer}:{base}", self.width, self.k, seed=layer)

    def _predict(self, layer: int, prev: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        counter = self.tables[layer].get(prev)
        if not counter:
            return None
        return tuple(max(counter.items(), key=lambda item: (item[1], item[0]))[0])

    def observe(self, states: Sequence[Tuple[int, ...]], learn: bool = True) -> Dict[str, float]:
        self.step_count += 1
        updates = 0
        errors = []
        suppressed_dims = 0.0
        total_dims = 0.0
        for layer, current in enumerate(states[: self.layers]):
            prev = self.prev_states[layer]
            error = 1.0
            if prev is not None:
                predicted = self._predict(layer, prev)
                if predicted is not None:
                    overlap = len(set(predicted) & set(current))
                    error = 1.0 - overlap / max(1.0, float(self.k))
                if learn and error > self.theta:
                    self.tables[layer][prev][tuple(current)] += 1.0
                    updates += 1
            self.error_history[layer].append(float(error))
            errors.append(float(error))
            suppressed_dims += max(0.0, 1.0 - error) * self.width
            total_dims += self.width
            self.prev_states[layer] = tuple(current)
        self.update_count += updates
        if updates == 0:
            self.zero_update_count += 1
        return {
            "mean_error": sum(errors) / max(1, len(errors)),
            "suppression": suppressed_dims / max(1.0, total_dims),
            "updates": float(updates),
        }

    def reset_state(self) -> None:
        self.prev_states = [None for _ in range(self.layers)]

    def diagnostics(self) -> Dict[str, float]:
        return {
            "steps": float(self.step_count),
            "updates": float(self.update_count),
            "zero_update_ratio": self.zero_update_count / max(1.0, float(self.step_count)),
            "mean_error": sum(sum(v) for v in self.error_history) / max(1, sum(len(v) for v in self.error_history)),
        }


class OnePassCausalMemory:
    """One-observation local causal and chain memory."""

    def __init__(self):
        self.direct: Dict[str, Counter] = defaultdict(Counter)

    def observe(self, cause: str, effect: str) -> None:
        self.direct[_norm(cause)][_norm(effect)] += 1.0

    def probability(self, cause: str, effect: str, vocab_size: int) -> float:
        counter = self.direct.get(_norm(cause), Counter())
        if not counter:
            return 1.0 / max(1.0, float(vocab_size))
        total = sum(counter.values())
        return float(counter.get(_norm(effect), 0.0)) / max(1.0, float(total))

    def chain(self, start: str, hops: int) -> Optional[str]:
        current = _norm(start)
        for _ in range(int(hops)):
            counter = self.direct.get(current)
            if not counter:
                return None
            current = max(counter.items(), key=lambda item: (item[1], item[0]))[0]
        return current


class VirtualSparseSDR:
    """Sparse SDR that returns active indices for very large dimensions."""

    def __init__(self, dim: int = 100000, k: int = 20, seed: int = 0):
        self.dim = int(dim)
        self.k = int(k)
        self.seed = int(seed)
        self.related: Dict[str, Counter] = defaultdict(Counter)
        self.bilingual = {
            "brain": "concept:brain",
            "não": "concept:brain",
            "nao": "concept:brain",
            "memory": "concept:memory",
            "ký ức": "concept:memory",
            "ky uc": "concept:memory",
        }

    def log2_capacity(self) -> float:
        n = float(self.dim)
        k = float(self.k)
        return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)) / math.log(2.0)

    def morphemes(self, word: str) -> List[str]:
        w = _norm(word).replace(" ", "_")
        parts = [f"word:{w}"]
        for prefix in ("un", "re", "pre", "post", "anti"):
            if w.startswith(prefix) and len(w) > len(prefix) + 2:
                parts.append(f"prefix:{prefix}")
                stem = w[len(prefix):]
                if stem.endswith("ness"):
                    stem = stem[:-4]
                if stem.endswith("i"):
                    stem = stem[:-1] + "y"
                parts.append(f"stem:{stem}")
        for suffix in ("ness", "ing", "ed", "er", "ly", "s"):
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                parts.append(f"suffix:{suffix}")
                stem = w[:-len(suffix)]
                if stem.endswith("i"):
                    stem = stem[:-1] + "y"
                if stem.startswith("un") and len(stem) > 4:
                    parts.append("prefix:un")
                    parts.append(f"stem:{stem[2:]}")
                parts.append(f"stem:{stem}")
        if w in self.bilingual:
            parts.append(self.bilingual[w])
        for other, count in self.related.get(w, Counter()).items():
            if count > 0:
                parts.append(f"related:{other}")
        return parts

    def observe_related(self, left: str, right: str) -> None:
        group = "pair:" + "|".join(sorted([_norm(left), _norm(right)]))
        self.related[_norm(left)][group] += 1.0
        self.related[_norm(right)][group] += 1.0

    def encode(self, word: str) -> Tuple[int, ...]:
        parts = self.morphemes(word)
        per = max(1, self.k // max(1, len(parts)))
        bits = []
        for part in parts:
            part_count = max(per, 3) if part.startswith(("prefix:", "stem:", "related:", "concept:")) else per
            bits.extend(_feature_bits(part, self.dim, part_count, self.seed))
        nonce = 0
        unique = []
        seen = set()
        for bit in bits:
            if bit not in seen:
                seen.add(bit)
                unique.append(bit)
        while len(unique) < self.k:
            bits.append(int(_hash_u64(self.seed, "fill", word, nonce) % self.dim))
            if bits[-1] not in seen:
                seen.add(bits[-1])
                unique.append(bits[-1])
            nonce += 1
        return tuple(sorted(unique[: self.k]))

    def overlap(self, left: str, right: str) -> int:
        return len(set(self.encode(left)) & set(self.encode(right)))


class HippocampalMemory:
    """Fast sparse auto-associative memory with slow consolidation."""

    def __init__(self, candidate_cap: int = 64, surprise_threshold: float = 0.5):
        self.candidate_cap = int(candidate_cap)
        self.surprise_threshold = float(surprise_threshold)
        self.fast: Dict[Tuple[str, ...], str] = {}
        self.slow: Dict[Tuple[str, ...], str] = {}
        self.feature_buckets: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
        self.encoded_count = 0
        self.seen_count = 0
        self.replay_count = 0
        self.last_candidate_count = 0
        self.last_full_scan = False

    def observe(self, features: Iterable[str], value: str, surprise: float = 1.0) -> bool:
        self.seen_count += 1
        key = tuple(sorted(_norm(f) for f in features if str(f).strip()))
        if not key or float(surprise) <= self.surprise_threshold:
            return False
        if key not in self.fast and key not in self.slow:
            for feature in key:
                self.feature_buckets[feature].append(key)
        self.fast[key] = _norm(value)
        self.encoded_count += 1
        return True

    def consolidate(self, replay_fraction: float = 0.10) -> int:
        budget = max(1, int(len(self.fast) * float(replay_fraction))) if self.fast else 0
        for key in list(self.fast.keys())[:budget]:
            self.slow[key] = self.fast[key]
            self.replay_count += 1
        return budget

    def recall(self, cue_features: Iterable[str]) -> Optional[str]:
        cues = tuple(sorted(_norm(f) for f in cue_features if str(f).strip()))
        if not cues:
            self.last_candidate_count = 0
            self.last_full_scan = False
            return None
        exact = self.slow.get(cues, self.fast.get(cues))
        if exact is not None:
            self.last_candidate_count = 1
            self.last_full_scan = False
            return exact
        candidates = []
        seen = set()
        for feature in cues:
            for key in reversed(self.feature_buckets.get(feature, [])):
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(key)
                if len(candidates) >= self.candidate_cap:
                    break
            if len(candidates) >= self.candidate_cap:
                break
        self.last_candidate_count = len(candidates)
        self.last_full_scan = False
        if not candidates:
            return None
        cue_set = set(cues)
        best = max(candidates, key=lambda key: (len(set(key) & cue_set), key))
        return self.slow.get(best, self.fast.get(best))

    def diagnostics(self) -> Dict[str, float]:
        return {
            "fast": float(len(self.fast)),
            "slow": float(len(self.slow)),
            "encoded": float(self.encoded_count),
            "seen": float(self.seen_count),
            "replay_budget": self.replay_count / max(1.0, float(self.seen_count)),
            "last_candidate_count": float(self.last_candidate_count),
            "last_full_scan": float(self.last_full_scan),
        }


class BioNeuromodulator:
    """Dopamine/acetylcholine/serotonin style local update gates."""

    def __init__(self, novelty_window: int = 256):
        self.recent = deque(maxlen=int(novelty_window))
        self.seen = Counter()
        self.update_count = 0
        self.novel_update_count = 0
        self.weight_norm = 1.0
        self.sparsity = 0.02
        self.formal_count = 0
        self.casual_count = 0

    def gates(self, token: str, surprise: float, reward: float = 0.0) -> Dict[str, float]:
        token = _norm(token)
        novelty = 1.0 if self.seen[token] == 0 else 1.0 / (1.0 + self.seen[token])
        dopamine = min(1.0, max(0.0, 0.60 * float(surprise) + 0.35 * novelty + 0.25 * float(reward)))
        acetylcholine = min(1.0, max(0.0, 0.75 * novelty + 0.25 * float(surprise)))
        serotonin = min(1.0, max(0.0, 1.0 - abs(self.weight_norm - 1.0)))
        return {"dopamine": dopamine, "acetylcholine": acetylcholine, "serotonin": serotonin, "novelty": novelty}

    def observe(self, token: str, surprise: float, reward: float = 0.0) -> bool:
        token = _norm(token)
        g = self.gates(token, surprise, reward)
        should_update = g["dopamine"] * g["acetylcholine"] > 0.30
        if should_update:
            self.update_count += 1
            if self.seen[token] == 0 or surprise > 0.7:
                self.novel_update_count += 1
        self.weight_norm += 0.02 * (g["dopamine"] - 0.5) - 0.04 * (self.weight_norm - 1.0)
        self.sparsity += 0.002 * (g["acetylcholine"] - 0.5) - 0.05 * (self.sparsity - 0.02)
        self.weight_norm = min(1.10, max(0.90, self.weight_norm))
        self.sparsity = min(0.022, max(0.018, self.sparsity))
        if token in {"therefore", "please", "regards", "sincerely"}:
            self.formal_count += 1
        if token in {"hey", "cool", "thanks", "yep"}:
            self.casual_count += 1
        self.seen[token] += 1
        self.recent.append(token)
        return should_update

    def tone(self) -> str:
        return "formal" if self.formal_count >= self.casual_count else "casual"

    def curiosity_pick(self, candidates: Sequence[Tuple[str, float]]) -> str:
        return max(candidates, key=lambda item: (float(item[1]), _norm(item[0])))[0]

    def diagnostics(self) -> Dict[str, float]:
        return {
            "updates": float(self.update_count),
            "novel_update_ratio": self.novel_update_count / max(1.0, float(self.update_count)),
            "weight_norm": float(self.weight_norm),
            "sparsity": float(self.sparsity),
        }


@dataclass
class DendriticSegment:
    active_bits: Tuple[int, ...]
    output: int
    threshold: float
    strength: float = 1.0
    weights: Tuple[float, ...] = ()
    branch_id: int = 0
    last_activation: float = 0.0
    local_update_count: int = 0

    def __post_init__(self) -> None:
        self.active_bits = tuple(sorted({int(bit) for bit in self.active_bits}))
        if not self.weights:
            self.weights = tuple(1.0 for _ in self.active_bits)
        else:
            self.weights = tuple(float(weight) for weight in self.weights[: len(self.active_bits)])
            if len(self.weights) < len(self.active_bits):
                self.weights = self.weights + tuple(1.0 for _ in range(len(self.active_bits) - len(self.weights)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 60.0:
            return 1.0
        if value <= -60.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-float(value)))

    def flat_drive(self, bits: Iterable[int]) -> float:
        active = {int(bit) for bit in bits}
        return float(sum(weight for bit, weight in zip(self.active_bits, self.weights) if bit in active))

    def activation(self, bits: Iterable[int]) -> float:
        self.last_activation = self._sigmoid(self.flat_drive(bits) - float(self.threshold))
        return float(self.last_activation)

    def spike(self, bits: Iterable[int]) -> bool:
        return self.activation(bits) >= 0.5

    def hebbian_update(self, bits: Iterable[int], lr: float = 0.05, max_weight: float = 1.0) -> int:
        active = {int(bit) for bit in bits}
        branch_activation = self.activation(bits)
        if branch_activation < 0.5:
            return 0
        updates = 0
        new_weights = []
        for bit, weight in zip(self.active_bits, self.weights):
            if bit in active and weight < max_weight - 1e-9:
                weight = min(float(max_weight), float(weight) + float(lr) * branch_activation * (float(max_weight) - float(weight)))
                updates += 1
            new_weights.append(float(weight))
        if updates:
            self.weights = tuple(new_weights)
            self.local_update_count += 1
        return updates


class DendriticLayer:
    """Sparse nonlinear dendritic tree; each branch is a local mini-processor."""

    def __init__(
        self,
        input_dim: int = 1024,
        outputs: int = 2,
        segment_size: int = 3,
        branches_per_output: int = 0,
        branch_size: Optional[int] = None,
        soma_threshold: float = 0.5,
        seed: int = 0,
    ):
        self.input_dim = int(input_dim)
        self.outputs = int(outputs)
        self.segment_size = int(segment_size)
        self.branch_size = int(branch_size or segment_size)
        self.soma_threshold = float(soma_threshold)
        self.seed = int(seed)
        self.branches: List[DendriticSegment] = []
        self.segments = self.branches
        self.last_ops = 0
        self.last_active_branches = 0
        self.last_updated_branches = 0
        self.branch_local_update_events = 0
        self.global_error_updates = 0
        if int(branches_per_output) > 0:
            self.initialize_tree(branches_per_output=int(branches_per_output), branch_size=self.branch_size)

    def initialize_tree(self, branches_per_output: int = 1000, branch_size: Optional[int] = None) -> None:
        size = int(branch_size or self.branch_size)
        cursor = 0
        for output in range(max(1, self.outputs)):
            for _ in range(int(branches_per_output)):
                bits = tuple((cursor + offset) % self.input_dim for offset in range(size))
                cursor += size
                self.add_branch(bits, output=output, threshold=max(0.5, float(size) - 0.5))

    def add_branch(
        self,
        bits: Iterable[int],
        output: int = 0,
        threshold: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        strength: float = 1.0,
    ) -> DendriticSegment:
        selected = tuple(sorted({int(bit) % self.input_dim for bit in bits}))
        if not selected:
            raise ValueError("Dendritic branch needs at least one active bit")
        branch = DendriticSegment(
            selected,
            int(output),
            float(threshold if threshold is not None else max(0.5, len(selected) - 0.5)),
            strength=float(strength),
            weights=tuple(weights or ()),
            branch_id=len(self.branches),
        )
        self.branches.append(branch)
        return branch

    def observe(self, bits: Iterable[int], output: int) -> None:
        selected = tuple(sorted(int(b) % self.input_dim for b in bits))[: self.segment_size]
        if len(selected) < self.segment_size:
            return
        self.add_branch(selected, output=int(output), threshold=max(0.5, float(len(selected)) - 0.5))

    def observe_or(self, bits: Iterable[int], output: int) -> None:
        selected = tuple(sorted(int(b) % self.input_dim for b in bits))[: self.segment_size]
        if selected:
            self.add_branch(selected, output=int(output), threshold=0.5)

    def branch_activations(self, bits: Iterable[int], output: Optional[int] = None) -> List[float]:
        activations = []
        self.last_ops = 0
        self.last_active_branches = 0
        for branch in self.branches:
            if output is not None and int(branch.output) != int(output):
                continue
            self.last_ops += len(branch.active_bits)
            activation = branch.activation(bits)
            activations.append(activation)
            if activation >= 0.5:
                self.last_active_branches += 1
        return activations

    def predict(self, bits: Iterable[int]) -> Optional[int]:
        votes = Counter()
        self.last_ops = 0
        self.last_active_branches = 0
        for branch in self.branches:
            self.last_ops += len(branch.active_bits)
            activation = branch.activation(bits)
            if activation >= 0.5:
                votes[branch.output] += branch.strength * activation
                self.last_active_branches += 1
        if not votes:
            return None
        return int(max(votes.items(), key=lambda item: (item[1], -item[0]))[0])

    def soma_activation(self, bits: Iterable[int], output: Optional[int] = None) -> float:
        activations = self.branch_activations(bits, output=output)
        suprathreshold = [activation for activation in activations if activation >= 0.5]
        return DendriticSegment._sigmoid(sum(suprathreshold) - self.soma_threshold)

    def soma_spike(self, bits: Iterable[int], output: Optional[int] = None) -> bool:
        return self.soma_activation(bits, output=output) >= 0.5

    def learn_branch_local(self, bits: Iterable[int], output: Optional[int] = None, lr: float = 0.05) -> int:
        self.last_ops = 0
        self.last_active_branches = 0
        self.last_updated_branches = 0
        update_ops = 0
        for branch in self.branches:
            if output is not None and int(branch.output) != int(output):
                continue
            self.last_ops += len(branch.active_bits)
            activation = branch.activation(bits)
            if activation >= 0.5:
                self.last_active_branches += 1
                updates = branch.hebbian_update(bits, lr=lr)
                if updates:
                    self.last_updated_branches += 1
                    self.branch_local_update_events += 1
                    update_ops += updates
        return update_ops

    def branch_activity_overlap(self, samples: Sequence[Iterable[int]]) -> float:
        active_sets = []
        for branch in self.branches:
            active = {
                idx
                for idx, sample in enumerate(samples)
                if branch.activation(sample) >= 0.5
            }
            if active:
                active_sets.append(active)
        if len(active_sets) < 2:
            return 0.0
        overlaps = []
        for i in range(len(active_sets)):
            for j in range(i + 1, len(active_sets)):
                union = active_sets[i] | active_sets[j]
                if union:
                    overlaps.append(len(active_sets[i] & active_sets[j]) / len(union))
        return sum(overlaps) / max(1, len(overlaps))

    def dense_ops_proxy(self) -> int:
        return max(1, self.input_dim * max(1, len(self.branches)) * max(1, self.outputs))

    def diagnostics(self) -> Dict[str, float]:
        branches = max(1, len(self.branches))
        mean_branch_size = sum(len(branch.active_bits) for branch in self.branches) / float(branches)
        return {
            "segments": float(len(self.branches)),
            "branches": float(len(self.branches)),
            "branches_per_neuron": float(len(self.branches)) / max(1.0, float(self.outputs)),
            "mean_branch_size": float(mean_branch_size),
            "last_active_branches": float(self.last_active_branches),
            "last_active_branch_ratio": float(self.last_active_branches) / float(branches),
            "last_updated_branches": float(self.last_updated_branches),
            "last_zero_update_branch_ratio": 1.0 - float(self.last_updated_branches) / float(branches),
            "branch_local_update_events": float(self.branch_local_update_events),
            "global_error_updates": float(self.global_error_updates),
            "last_ops": float(self.last_ops),
            "dense_ops_proxy": float(self.dense_ops_proxy()),
            "compute_density_gain": float(self.dense_ops_proxy()) / max(1.0, float(self.last_ops)),
        }


class BioComputeAgent(IntegratedLSLAgent):
    """Integrated Phase 9 agent with optional biological compute modules."""

    def __init__(
        self,
        vocab_size: int = 4000,
        tokenizer: str = "subword",
        candidate_cap: int = 128,
        seed: int = 0,
        use_pc_v2: bool = True,
        use_sdr_v2: bool = True,
        use_columns: bool = True,
        use_hippocampus: bool = True,
        use_dendrites: bool = True,
        use_neuromodulation: bool = True,
    ):
        super().__init__(vocab_size=vocab_size, tokenizer=tokenizer, candidate_cap=candidate_cap, seed=seed)
        self.use_pc_v2 = bool(use_pc_v2)
        self.use_sdr_v2 = bool(use_sdr_v2)
        self.use_columns = bool(use_columns)
        self.use_hippocampus = bool(use_hippocampus)
        self.use_dendrites = bool(use_dendrites)
        self.use_neuromodulation = bool(use_neuromodulation)
        self.pc_v2 = LocalPredictiveStack(layers=3, width=256, k=8, theta=0.05)
        self.sdr_v2 = VirtualSparseSDR(dim=100000, k=20, seed=seed)
        self.columns = CorticalColumnSequenceMemory(
            vocab_size=max(128, vocab_size),
            cells_per_column=100,
            sparsity=0.02,
            seed=seed,
        )
        self.hippocampus = HippocampalMemory(candidate_cap=candidate_cap)
        self.bio_modulator = BioNeuromodulator()
        self.dendrites = DendriticLayer(input_dim=2048, outputs=8, segment_size=3)
        self.sdr_observations = 0
        self.sdr_active_total = 0

    def build_tokenizer(self, text: str) -> None:
        super().build_tokenizer(text)
        if self.use_columns and self.columns.vocab_size < self.vocab_size:
            self.columns = CorticalColumnSequenceMemory(
                vocab_size=max(128, self.vocab_size),
                cells_per_column=100,
                sparsity=0.02,
                seed=self.seed,
            )

    def observe_text(self, text: str, source: str = "text", learn_transitions: bool = True) -> None:
        super().observe_text(text, source=source, learn_transitions=learn_transitions)
        tokens = self.tokenizer.encode(text) if learn_transitions else []
        for token in tokens:
            if self.use_sdr_v2:
                bits = self.sdr_v2.encode(str(token))
                self.sdr_observations += 1
                self.sdr_active_total += len(bits)
            if self.use_columns and int(token) < self.columns.vocab_size:
                self.columns.forward(int(token), learn=True)
            if self.use_pc_v2:
                states = [self.pc_v2.state_for(token + layer * 997, layer) for layer in range(self.pc_v2.layers)]
                self.pc_v2.observe(states, learn=True)
            if self.use_neuromodulation:
                self.bio_modulator.observe(str(token), surprise=0.8 if self.bio_modulator.seen[str(token)] == 0 else 0.05)

    def predict_next_token_id(self, prefix: Sequence[int]) -> Optional[int]:
        tokens = [int(token) for token in prefix]
        if not tokens:
            return None
        votes = Counter()
        if self.use_columns:
            self.columns.reset_state()
            for token in tokens:
                if int(token) < self.columns.vocab_size:
                    self.columns.forward(int(token), learn=False)
            scores = self.columns.predict_next_token_scores()
            if float(scores.sum()) > 0.0:
                top = int(scores.argmax())
                votes[top] += float(scores[top])
        self.long_context.reset_state()
        for token in tokens[:-1]:
            self.long_context.advance_context(int(token))
        remembered, confidence = self.long_context.predict_next(
            int(tokens[-1]),
            vocab_size=self.vocab_size,
            return_confidence=True,
            update_context=False,
        )
        if remembered is not None:
            votes[int(remembered)] += max(0.01, float(confidence))
        if not votes:
            return None
        return int(max(votes.items(), key=lambda item: (item[1], -item[0]))[0])

    def observe_fact(self, key: str, value: str, surprise: float = 1.0) -> None:
        self.observe_text(f"The value for {key} is {value}.", source="bio_fact", learn_transitions=False)
        if self.use_hippocampus:
            self.hippocampus.observe(["fact", key], value, surprise=surprise)

    def recall_fact(self, key: str) -> Optional[str]:
        if self.use_hippocampus:
            value = self.hippocampus.recall(["fact", key])
            if value is not None:
                return value
        return self.answer(f"What is the value for {key}?")

    def observe_context_pattern(self, bits: Iterable[int], output: int) -> None:
        if self.use_dendrites:
            self.dendrites.observe(bits, output)

    def predict_context_pattern(self, bits: Iterable[int]) -> Optional[int]:
        if not self.use_dendrites:
            return None
        return self.dendrites.predict(bits)

    def consolidate(self, replay_fraction: float = 0.10) -> int:
        if not self.use_hippocampus:
            return 0
        return self.hippocampus.consolidate(replay_fraction=replay_fraction)

    def diagnostics(self) -> Dict[str, float]:
        column_metrics = self.columns.metrics()
        return {
            **super().diagnostics(),
            **{f"pc_v2_{k}": v for k, v in self.pc_v2.diagnostics().items()},
            "sdr_v2_observations": float(self.sdr_observations),
            "sdr_v2_mean_active": self.sdr_active_total / max(1.0, float(self.sdr_observations)),
            **{f"column_{k}": float(v) for k, v in column_metrics.items()},
            **{f"hippocampus_{k}": v for k, v in self.hippocampus.diagnostics().items()},
            **{f"bio_mod_{k}": v for k, v in self.bio_modulator.diagnostics().items()},
            **{f"dendrite_{k}": v for k, v in self.dendrites.diagnostics().items()},
        }
