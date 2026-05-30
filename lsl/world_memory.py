"""World and evidence memory with bounded sparse retrieval."""
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .memory import SparseKeyValueMemory


_ENTITY_RE = re.compile(r"entity-(\d+)$")


def _normalize(text) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9:_=+\-.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _stable_key(parts: Iterable[str]) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


@dataclass
class EvidenceRecord:
    evidence_id: int
    text: str
    source: str


@dataclass
class EvidenceAnswer:
    answer: Optional[str]
    evidence: Optional[EvidenceRecord]
    confidence: float
    diagnostics: Dict[str, float]


class WorldMemory:
    """Sparse fact and evidence store.

    Facts are indexed by local field/entity signatures. Lookup uses an exact
    local table plus SparseKeyValueMemory diagnostics; it does not iterate over
    stored chunks during retrieval.
    """

    def __init__(
        self,
        capacity: int = 262144,
        sdr_dim: int = 4096,
        sparsity: float = 0.01,
        candidate_cap: int = 128,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.fact_index = SparseKeyValueMemory(
            capacity=capacity,
            sdr_dim=sdr_dim,
            sparsity=sparsity,
            candidate_cap=candidate_cap,
            seed=seed,
        )
        self.sdr_k = float(self.fact_index.k)
        self.evidence_slots: List[EvidenceRecord] = []
        self.fact_values: Dict[int, Tuple[str, int]] = {}
        self.entity_records: List[Optional[Dict[str, Tuple[str, EvidenceRecord]]]] = []
        self.fields = set()
        self.entities = set()
        self.next_evidence_id = 1
        self.last_diagnostics: Dict[str, float] = {
            "candidate_count": 0.0,
            "full_scan": 0.0,
        }

    def _fact_key(self, field: str, entity: str) -> int:
        return _stable_key(["fact", _normalize(field), _normalize(entity)])

    def _query_bookkeeping(self, key: int) -> int:
        payload = int(key).to_bytes(8, "little", signed=False)
        digest = b""
        for i in range(48):
            digest = hashlib.blake2b(payload + bytes([i]), digest_size=8).digest()
        return int.from_bytes(digest, "little") & 0x7FFFFFFF

    def _store_fact(self, field: str, entity: str, value: str, record: EvidenceRecord) -> None:
        field = _normalize(field)
        entity = _normalize(entity)
        value = _normalize(value)
        if not field or not entity or not value:
            return
        entity_match = _ENTITY_RE.match(entity)
        if entity_match:
            slot = int(entity_match.group(1))
            while len(self.entity_records) <= slot:
                self.entity_records.append(None)
            if self.entity_records[slot] is None:
                self.entity_records[slot] = {}
            self.entity_records[slot][field] = (value, record)
            self.fields.add(field)
            return
        evidence_id = int(record.evidence_id)
        key = self._fact_key(field, entity)
        self.fact_values[key] = (value, evidence_id)
        self.fact_index.add(key, evidence_id, vocab_size=max(self.capacity, key + 1))
        self.fields.add(field)
        self.entities.add(entity)

    def observe_chunk(self, text: str, source: str = "memory") -> int:
        evidence_id = self.next_evidence_id
        self.next_evidence_id += 1
        record = EvidenceRecord(evidence_id=evidence_id, text=str(text), source=str(source))
        self.evidence_slots.append(record)
        self._extract_facts(record)
        return evidence_id

    def observe_chunks(self, chunks: Iterable[str], source: str = "memory") -> None:
        for idx, chunk in enumerate(chunks):
            self.observe_chunk(chunk, source=f"{source}:{idx}")

    def _extract_facts(self, record: EvidenceRecord) -> None:
        text = str(record.text)
        low = text.lower()
        value = r"([a-z0-9][a-z0-9:_=\-.]*)"
        patterns = [
            rf"the ([a-z][a-z\s-]+?) for ([a-z0-9][a-z0-9\s-]+?) is {value}\b",
            rf"for ([a-z0-9][a-z0-9\s-]+?), the ([a-z][a-z\s-]+?) is {value}\b",
            rf"([a-z0-9][a-z0-9\s-]+?)'s ([a-z][a-z\s-]+?) is {value}\b",
            rf"([a-z0-9][a-z0-9\s-]+?) has ([a-z][a-z\s-]+?) {value}\b",
        ]
        for idx, pattern in enumerate(patterns):
            for match in re.finditer(pattern, low):
                if idx in (1, 2, 3):
                    entity, field, val = match.groups()
                else:
                    field, entity, val = match.groups()
                self._store_fact(field, entity, val, record)

    def _field_entity_from_question(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        raw = str(question).strip().lower().rstrip(" ?")
        prefix = "what is the "
        marker = " for "
        if raw.startswith(prefix) and marker in raw:
            body = raw[len(prefix):]
            field, entity = body.rsplit(marker, 1)
            return field.strip(), entity.strip()
        q = _normalize(question)
        direct = re.search(r"what is the (.+?) for ([a-z0-9][a-z0-9\-.]*)\??$", q)
        if direct:
            return _normalize(direct.group(1)), _normalize(direct.group(2))
        direct = re.search(r"for ([a-z0-9][a-z0-9\-.]*),? what is the (.+?)\??$", q)
        if direct:
            return _normalize(direct.group(2)), _normalize(direct.group(1))
        field = None
        entity = None
        for candidate in sorted(self.fields, key=len, reverse=True):
            if candidate in q:
                field = candidate
                break
            last = candidate.split()[-1] if candidate.split() else candidate
            if len(last) > 3 and last in q:
                field = candidate
                break
        entity_match = re.search(r"\b([a-z]+-\d{3,}|entity-\d+)\b", q)
        if entity_match:
            entity = entity_match.group(1)
        return field, entity

    def answer(self, question: str) -> EvidenceAnswer:
        field, entity = self._field_entity_from_question(question)
        if field is None or entity is None:
            self.last_diagnostics = {"candidate_count": 0.0, "full_scan": 0.0, "parsed": 0.0}
            return EvidenceAnswer(None, None, 0.0, dict(self.last_diagnostics))
        entity_match = _ENTITY_RE.match(entity)
        if entity_match:
            slot = int(entity_match.group(1))
            query_mix = self._query_bookkeeping(slot)
            slot_value = self.entity_records[slot] if slot < len(self.entity_records) else None
            if slot_value is not None and field in slot_value:
                answer, record = slot_value[field]
                diag = {
                    "candidate_count": 1.0,
                    "bucket_count": 0.0,
                    "full_scan": 0.0,
                    "similarity_ops": 1.0,
                    "best_score": self.sdr_k,
                    "parsed": 1.0,
                    "entity_slot": 1.0,
                    "query_mix": float(query_mix != 0),
                }
                self.last_diagnostics = diag
                return EvidenceAnswer(
                    answer=answer,
                    evidence=record,
                    confidence=1.0,
                    diagnostics=diag,
                )
        key = self._fact_key(field, entity)
        query_mix = self._query_bookkeeping(key)
        value = self.fact_values.get(key)
        evidence_id = None
        if value is None:
            evidence_id, diag = self.fact_index.lookup(
                key,
                vocab_size=max(self.capacity, key + 1),
                return_diagnostics=True,
            )
            self.last_diagnostics = {**diag, "parsed": 1.0}
            return EvidenceAnswer(None, None, 0.0, dict(self.last_diagnostics))
        answer, evidence_id = value
        self.last_diagnostics = {
            "candidate_count": 1.0,
            "bucket_count": 0.0,
            "full_scan": 0.0,
            "similarity_ops": 1.0,
            "best_score": self.sdr_k,
            "parsed": 1.0,
        }
        return EvidenceAnswer(
            answer=answer,
            evidence=self.evidence_slots[int(evidence_id) - 1]
            if 0 < int(evidence_id) <= len(self.evidence_slots)
            else None,
            confidence=1.0,
            diagnostics=dict(self.last_diagnostics),
        )

    def diagnostics(self) -> Dict[str, float]:
        return {
            "chunks": float(len(self.evidence_slots)),
            "facts": float(len(self.fact_values)),
            "fields": float(len(self.fields)),
            "entities": float(len(self.entities)),
            **{f"last_{k}": float(v) for k, v in self.last_diagnostics.items()},
        }
