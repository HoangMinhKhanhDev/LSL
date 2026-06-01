"""Shared multilingual semantic aliases for SDR-style encoders."""
from __future__ import annotations

from typing import Dict, Tuple

from .text_normalization import lexical_key


MULTILINGUAL_CONCEPT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "neuromedical": (
        "brain",
        "neuron",
        "neurons",
        "synapse",
        "synapses",
        "cortex",
        "neural",
        "plasticity",
        "não",
        "não bộ",
        "nơron",
        "vỏ não",
        "não người",
    ),
    "language": (
        "language",
        "speech",
        "speaking",
        "reading",
        "writing",
        "communication",
        "comprehension",
        "production",
        "words",
        "recovery",
        "ngôn ngữ",
        "lời nói",
        "đọc",
        "viết",
        "giao tiếp",
    ),
    "care": (
        "therapy",
        "doctor",
        "patient",
        "patients",
        "treatment",
        "medication",
        "examined",
        "prescribed",
        "session",
        "sessions",
        "điều trị",
        "bác sĩ",
        "bệnh nhân",
        "thuốc",
    ),
    "learning": (
        "memory",
        "learning",
        "learn",
        "patterns",
        "cognitive",
        "thinking",
        "attention",
        "perception",
        "processing",
        "activity",
        "ký ức",
        "học",
        "học tập",
        "nhận thức",
        "chú ý",
    ),
    "body": (
        "heart",
        "blood",
        "pressure",
        "muscle",
        "diabetes",
        "sugar",
        "flow",
        "cơ thể",
        "tim",
        "máu",
        "áp lực",
        "cơ",
        "đường",
    ),
    "environment": (
        "river",
        "water",
        "sky",
        "park",
        "bed",
        "worldwide",
        "môi trường",
        "sông",
        "nước",
        "bầu trời",
        "công viên",
        "thế giới",
    ),
    "object": (
        "table",
        "ball",
        "mouse",
        "stranger",
        "fish",
        "meat",
        "đối tượng",
        "bàn",
        "bóng",
        "chuột",
        "cá",
        "thịt",
    ),
}


MULTILINGUAL_TRANSLATION_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("brain", "não", "neuromedical"),
    ("neuron", "nơron", "neuromedical"),
    ("cortex", "vỏ não", "neuromedical"),
    ("memory", "ký ức", "learning"),
    ("learning", "học tập", "learning"),
    ("language", "ngôn ngữ", "language"),
    ("speech", "lời nói", "language"),
    ("therapy", "điều trị", "care"),
    ("doctor", "bác sĩ", "care"),
    ("patient", "bệnh nhân", "care"),
    ("body", "cơ thể", "body"),
    ("environment", "môi trường", "environment"),
    ("object", "đối tượng", "object"),
)


def group_for_word(word: str) -> str | None:
    return ALIAS_TO_GROUP.get(lexical_key(word))


def word_variants_for_group(group: str) -> Tuple[str, ...]:
    return MULTILINGUAL_CONCEPT_ALIASES.get(group, ())


def all_multilingual_terms() -> Tuple[str, ...]:
    terms = []
    seen = set()
    for words in MULTILINGUAL_CONCEPT_ALIASES.values():
        for word in words:
            key = lexical_key(word)
            if key not in seen:
                seen.add(key)
                terms.append(word)
    return tuple(terms)


def alias_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group, words in MULTILINGUAL_CONCEPT_ALIASES.items():
        for word in words:
            mapping[lexical_key(word)] = group
    return mapping


ALIAS_TO_GROUP = alias_map()
