"""Sparse local workspace and entity-event graph for compositional reasoning."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class WorkspaceStep:
    name: str
    value: int
    support: float


class ReasoningWorkspace:
    """Bounded local workspace for steps, variables, bindings, and subgoals."""

    def __init__(self, capacity: int = 512):
        self.capacity = int(capacity)
        self.variables: Dict[str, int] = {}
        self.bindings: Dict[Tuple[int, int], int] = {}
        self.steps = deque(maxlen=self.capacity)
        self.subgoals = deque(maxlen=self.capacity)

    def bind(self, role: str, filler: int) -> None:
        self.variables[str(role)] = int(filler)

    def resolve(self, role: str) -> Optional[int]:
        value = self.variables.get(str(role))
        return None if value is None else int(value)

    def bind_pair(self, left: int, relation: int, right: int) -> None:
        self.bindings[(int(left), int(relation))] = int(right)

    def resolve_pair(self, left: int, relation: int) -> Optional[int]:
        value = self.bindings.get((int(left), int(relation)))
        return None if value is None else int(value)

    def add_step(self, name: str, value: int, support: float = 1.0) -> None:
        self.steps.append(WorkspaceStep(str(name), int(value), float(support)))

    def add_subgoal(self, name: str, value: int) -> None:
        self.subgoals.append((str(name), int(value)))

    def clear(self) -> None:
        self.variables.clear()
        self.bindings.clear()
        self.steps.clear()
        self.subgoals.clear()

    def diagnostics(self) -> Dict[str, float]:
        return {
            "variables": float(len(self.variables)),
            "bindings": float(len(self.bindings)),
            "steps": float(len(self.steps)),
            "subgoals": float(len(self.subgoals)),
        }


class EntityEventGraph:
    """Local graph for entity, event, episode, and evidence reasoning."""

    def __init__(self, candidate_cap: int = 64):
        self.candidate_cap = int(candidate_cap)
        self.edges: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self.relation_shards: Dict[int, Dict[int, List[Optional[int]]]] = defaultdict(dict)
        self.evidence: Dict[Tuple[int, int, int], int] = {}
        self.episodes: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.last_candidate_count = 0
        self.last_full_scan = False

    def _local_verify(self, subject: int, relation: int) -> int:
        x = (int(subject) ^ (int(relation) << 11)) & 0xFFFFFFFF
        for _ in range(6):
            x ^= (x << 7) & 0xFFFFFFFF
            x ^= x >> 9
        return int(x)

    def observe_event(
        self,
        subject: int,
        relation: int,
        obj: int,
        episode_id: int = 0,
        evidence_id: int = 0,
        strength: float = 1.0,
    ) -> None:
        s, r, o = int(subject), int(relation), int(obj)
        self.edges[(s, r)][o] += float(strength)
        shard_id = s >> 12
        offset = s & 4095
        shards = self.relation_shards[r]
        shard = shards.get(shard_id)
        if shard is None:
            shard = [None] * 4096
            shards[shard_id] = shard
        shard[offset] = o
        self.evidence[(s, r, o)] = int(evidence_id)
        self.episodes[int(episode_id)].append((s, r, o))

    def query(self, subject: int, relation: int) -> Optional[int]:
        subject = int(subject)
        relation = int(relation)
        self._local_verify(subject, relation)
        shard = self.relation_shards.get(relation, {}).get(subject >> 12)
        if shard is not None:
            value = shard[subject & 4095]
            if value is not None:
                self.last_full_scan = False
                self.last_candidate_count = 1
                return int(value)
        bucket = self.edges.get((subject, relation))
        self.last_full_scan = False
        self.last_candidate_count = 0 if not bucket else min(len(bucket), self.candidate_cap)
        if not bucket:
            return None
        return int(max(bucket.items(), key=lambda item: (item[1], -item[0]))[0])

    def query_chain(self, start: int, relations: Iterable[int]) -> Optional[int]:
        current = int(start)
        for relation in relations:
            nxt = self.query(current, int(relation))
            if nxt is None:
                return None
            current = int(nxt)
        return current

    def evidence_for(self, subject: int, relation: int, obj: int) -> Optional[int]:
        value = self.evidence.get((int(subject), int(relation), int(obj)))
        return None if value is None else int(value)

    def diagnostics(self) -> Dict[str, float]:
        return {
            "edge_keys": float(len(self.edges)),
            "episodes": float(len(self.episodes)),
            "last_candidate_count": float(self.last_candidate_count),
            "last_full_scan": float(self.last_full_scan),
        }
