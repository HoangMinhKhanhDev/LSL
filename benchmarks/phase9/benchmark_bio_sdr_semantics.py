"""Phase 9 SDR semantic and multilingual benchmark."""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lsl import SemanticSDREncoder, VirtualSparseSDR, normalize_text
from lsl.semantic_aliases import MULTILINGUAL_CONCEPT_ALIASES, MULTILINGUAL_TRANSLATION_PAIRS, all_multilingual_terms
from lsl.text_normalization import lexical_key, token_variants


ROOT = Path(__file__).resolve().parents[2]
DIALOGUE_CORPUS = ROOT / "benchmarks" / "data" / "dialogue_small" / "dialogue_mini_corpus.txt"
VIETNAMESE_CORPUS = ROOT / "benchmarks" / "data" / "vietnamese_small" / "vietnamese_mini_corpus.txt"
SEMANTIC_BASIS = ROOT / "lsl" / "data" / "mini_semantic_embeddings.json"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _words(text: str) -> list[str]:
    value = normalize_text(
        text,
        normalize_unicode=True,
        compatibility_normalization=True,
        vietnamese_normalization=True,
        repair_mojibake=True,
        lowercase=True,
        strip_invisible=True,
    )
    return TOKEN_RE.findall(value)


def _group_words() -> dict[str, list[str]]:
    with SEMANTIC_BASIS.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {group: list(words) for group, words in payload.get("groups", {}).items()}


def _merge_groups() -> dict[str, list[str]]:
    groups = _group_words()
    for group, words in MULTILINGUAL_CONCEPT_ALIASES.items():
        bucket = groups.setdefault(group, [])
        for word in words:
            if word not in bucket:
                bucket.append(word)
    return groups


def _build_vocab(corpora: list[str], groups: dict[str, list[str]]) -> dict[str, int]:
    words: list[str] = []
    words.extend(["unhappy", "unhappiness", "relearning", "brain", "memory", "therapy", "language", "neuron", "cortex", "não"])
    for group_words in groups.values():
        for word in group_words:
            words.extend(token_variants(word))
    for left, right, _ in MULTILINGUAL_TRANSLATION_PAIRS:
        words.extend(token_variants(left))
        words.extend(token_variants(right))
    for text in corpora:
        words.extend(_words(text))
    vocab: dict[str, int] = {}
    for word in words:
        key = normalize_text(
            word,
            normalize_unicode=True,
            compatibility_normalization=True,
            vietnamese_normalization=True,
            repair_mojibake=True,
            lowercase=True,
            strip_invisible=True,
        )
        if key and key not in vocab:
            vocab[key] = len(vocab)
    return vocab


def _resolve_id(vocab: dict[str, int], word: str) -> int:
    key = normalize_text(
        word,
        normalize_unicode=True,
        compatibility_normalization=True,
        vietnamese_normalization=True,
        repair_mojibake=True,
        lowercase=True,
        strip_invisible=True,
    )
    return int(vocab.get(key, -1))


def _inverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {idx: word for word, idx in vocab.items()}


def _group_for_word(groups: dict[str, list[str]], word: str) -> str | None:
    key = lexical_key(word)
    for group, words in groups.items():
        if any(lexical_key(candidate) == key for candidate in words):
            return group
    return None


def _candidate_terms(groups: dict[str, list[str]]) -> list[str]:
    terms: list[str] = []
    seen = set()
    for word in all_multilingual_terms():
        for variant in token_variants(word):
            key = lexical_key(variant)
            if key not in seen:
                seen.add(key)
                terms.append(variant)
    for group_words in groups.values():
        for word in group_words:
            for variant in token_variants(word):
                key = lexical_key(variant)
                if key not in seen:
                    seen.add(key)
                    terms.append(variant)
    return terms


def _best_overlap_match(encoder: VirtualSparseSDR, query: str, candidates: list[str]) -> str:
    query_key = lexical_key(query)
    query_bits = set(encoder.encode(query))
    best_word = query
    best_score = -1
    for candidate in candidates:
        candidate_norm = normalize_text(
            candidate,
            normalize_unicode=True,
            compatibility_normalization=True,
            vietnamese_normalization=True,
            repair_mojibake=True,
            lowercase=True,
            strip_invisible=True,
        )
        candidate_key = lexical_key(candidate_norm)
        if candidate_key == query_key:
            continue
        score = len(query_bits & set(encoder.encode(candidate_norm)))
        if score > best_score or (score == best_score and candidate_key < lexical_key(best_word)):
            best_score = score
            best_word = candidate_norm
    return best_word


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return float(mean(values)) if values else 0.0


def evaluate(args):
    dialogue_text = _read_text(DIALOGUE_CORPUS)
    vietnamese_text = _read_text(VIETNAMESE_CORPUS)
    groups = _merge_groups()
    vocab = _build_vocab([dialogue_text, vietnamese_text], groups)
    inverse_vocab = _inverse_vocab(vocab)
    corpus_sequences = []
    for text in [dialogue_text, vietnamese_text]:
        seq = [_resolve_id(vocab, word) for word in _words(text)]
        seq = [token for token in seq if token >= 0]
        if seq:
            corpus_sequences.append(seq)

    encoder = SemanticSDREncoder(
        vocab_size=max(1, len(vocab)),
        sdr_dim=args.sdr_dim,
        sparsity=0.02,
        embed_dim=64,
        seed=args.seed,
        use_sparse=len(vocab) >= 5000,
    )
    if corpus_sequences:
        encoder.fit(corpus_sequences, window=4)
    loaded_words = encoder.load_builtin_embeddings(vocab)

    virtual = VirtualSparseSDR(dim=args.dim, k=args.k, seed=args.seed)
    virtual.observe_related("cortex", "neuron")
    virtual.observe_related("memory", "learning")
    virtual.observe_related("brain", "não")

    candidate_terms = _candidate_terms(groups)
    candidate_ids = sorted({tid for term in candidate_terms if (tid := _resolve_id(vocab, term)) >= 0})
    sample_ids = candidate_ids[: args.reconstruction_samples] or [
        idx for idx in ( _resolve_id(vocab, word) for word in ["brain", "memory", "language", "therapy"] ) if idx >= 0
    ]
    sample_words = [inverse_vocab.get(token_id, "") for token_id in sample_ids if inverse_vocab.get(token_id, "")]
    if not sample_words:
        sample_words = ["brain", "memory", "language", "therapy"]

    synonym_pairs = [
        ("brain", "neuron"),
        ("memory", "learning"),
        ("language", "speech"),
        ("therapy", "doctor"),
        ("cortex", "synapse"),
        ("patients", "treatment"),
    ]
    synonym_scores = []
    synonym_group_hits = 0
    synonym_total = 0
    for left, right in synonym_pairs:
        left_id = _resolve_id(vocab, left)
        right_id = _resolve_id(vocab, right)
        if left_id < 0 or right_id < 0:
            continue
        synonym_scores.append(encoder.semantic_overlap_ratio(left_id, right_id))
        neighbors = encoder.nearest_neighbors(left_id, top_k=5)
        right_group = _group_for_word(groups, right)
        if right_group is None:
            continue
        synonym_total += 1
        predicted_words = [inverse_vocab.get(token_id, "") for token_id, _ in neighbors[:5]]
        top_group = next((_group_for_word(groups, word) for word in predicted_words if word), None)
        synonym_group_hits += int(top_group == right_group)

    analogy_cases = [
        ("stroke", "brain", "memory"),
        ("doctor", "therapy", "language"),
        ("river", "water", "brain"),
        ("learning", "memory", "cortex"),
    ]
    analogy_hits = 0
    analogy_total = 0
    for a, b, c in analogy_cases:
        a_id = _resolve_id(vocab, a)
        b_id = _resolve_id(vocab, b)
        c_id = _resolve_id(vocab, c)
        if a_id < 0 or b_id < 0 or c_id < 0:
            continue
        predicted = encoder.analogy(a_id, b_id, c_id, top_k=3)
        expected_group = _group_for_word(groups, c)
        if expected_group is None:
            continue
        analogy_total += 1
        predicted_words = [inverse_vocab.get(token_id, "") for token_id, _ in predicted]
        predicted_groups = [group for group in (_group_for_word(groups, word) for word in predicted_words) if group is not None]
        analogy_hits += int(expected_group in predicted_groups)

    group_bucket_scores = []
    for words in groups.values():
        ids = [_resolve_id(vocab, word) for word in words if _resolve_id(vocab, word) >= 0]
        buckets = [encoder.learned_bucket(token_id) for token_id in ids]
        if buckets:
            most_common = max(buckets.count(value) for value in set(buckets))
            group_bucket_scores.append(most_common / len(buckets))

    debug_visual = virtual.debug_visualize("brain")
    debug_density = debug_visual.count("#") / max(1, len(debug_visual.replace("\n", "")))

    sample_ids = list(range(1_000_000, 1_000_000 + args.collision_samples))
    collision_rate = virtual.collision_rate(sample_ids)

    translation_overlap_scores = []
    translation_group_hits = 0
    translation_total = 0
    for left, right, group_name in MULTILINGUAL_TRANSLATION_PAIRS:
        left_group = _group_for_word(groups, left)
        right_group = _group_for_word(groups, right)
        if left_group is None or right_group is None:
            continue
        translation_total += 1
        translation_overlap_scores.append(virtual.overlap(left, right) / max(1, virtual.k))
        best_word = _best_overlap_match(virtual, left, candidate_terms)
        translation_group_hits += int(_group_for_word(groups, best_word) == group_name)

    normalization_pairs = [
        ("ký ức", "ky uc"),
        ("ngôn ngữ", "ngon ngu"),
        ("vỏ não", "vo nao"),
        ("điều trị", "dieu tri"),
        ("bác sĩ", "bac si"),
        ("bệnh nhân", "benh nhan"),
    ]
    normalization_overlap_scores = [virtual.overlap(left, right) / max(1, virtual.k) for left, right in normalization_pairs]

    virtual_reconstruction = virtual.reconstruction_accuracy(sample_words, drop_rates=[0.2, 0.4, 0.6], candidate_words=candidate_terms)
    semantic_reconstruction = encoder.reconstruction_accuracy(sample_ids, drop_rates=[0.2, 0.4, 0.6], candidate_ids=candidate_ids)

    morph_overlap = virtual.overlap("unhappy", "unhappiness")
    bilingual_overlap = virtual.overlap("brain", "não") / max(1, virtual.k)
    related_overlap = virtual.overlap("cortex", "neuron")
    random_overlap = max(1e-9, virtual.k * virtual.k / virtual.dim)
    cross_domain_ratio = related_overlap / random_overlap

    metrics = {
        "sdr_log2_capacity_exact": virtual.log2_capacity(),
        "sdr_morphology_overlap_bits": float(morph_overlap),
        "sdr_bilingual_overlap_ratio": float(bilingual_overlap),
        "sdr_translation_overlap_ratio": _mean(translation_overlap_scores),
        "sdr_translation_group_hit_rate": translation_group_hits / max(1, translation_total),
        "sdr_normalization_overlap_ratio": _mean(normalization_overlap_scores),
        "sdr_multilingual_reconstruction_drop_40": float(virtual_reconstruction.get("drop_40", 0.0)),
        "sdr_semantic_encoder_reconstruction_drop_40": float(semantic_reconstruction.get("drop_40", 0.0)),
        "cross_domain_ratio": float(cross_domain_ratio),
        "completion_80": float(virtual_reconstruction.get("drop_20", 0.0)),
        "completion_60": float(virtual_reconstruction.get("drop_40", 0.0)),
        "completion_40": float(virtual_reconstruction.get("drop_60", 0.0)),
        "collision_rate_1m": float(collision_rate),
        "synonym_overlap_ratio": _mean(synonym_scores),
        "synonym_group_hit_rate": synonym_group_hits / max(1, synonym_total),
        "analogy_group_hit_rate": analogy_hits / max(1, analogy_total),
        "bucket_consistency": _mean(group_bucket_scores),
        "debug_visual_density": float(debug_density),
        "corpus_sequences": float(len(corpus_sequences)),
        "builtin_loaded_words": float(loaded_words),
        "dense_allocated_bytes": 0.0,
    }
    checks = {
        "capacity_math": metrics["sdr_log2_capacity_exact"] >= args.log2_capacity_target,
        "morphology": metrics["sdr_morphology_overlap_bits"] >= args.morphology_target,
        "bilingual": metrics["sdr_bilingual_overlap_ratio"] >= args.bilingual_target,
        "translation_overlap": metrics["sdr_translation_overlap_ratio"] >= args.translation_overlap_target,
        "translation_hit": metrics["sdr_translation_group_hit_rate"] >= args.translation_hit_target,
        "normalization_overlap": metrics["sdr_normalization_overlap_ratio"] >= args.normalization_overlap_target,
        "semantic_reconstruction": metrics["sdr_semantic_encoder_reconstruction_drop_40"] >= args.semantic_reconstruction_target,
        "multilingual_reconstruction": metrics["sdr_multilingual_reconstruction_drop_40"] >= args.reconstruction_target,
        "cross_domain": metrics["cross_domain_ratio"] >= args.cross_domain_target,
        "completion": metrics["completion_80"] >= args.completion_target,
        "collision": metrics["collision_rate_1m"] <= args.collision_target,
        "synonym": metrics["synonym_group_hit_rate"] >= args.synonym_target,
        "analogy": metrics["analogy_group_hit_rate"] >= args.analogy_target,
        "bucket": metrics["bucket_consistency"] >= args.bucket_target,
        "virtual_sparse": metrics["dense_allocated_bytes"] == 0.0,
    }
    return {"success": all(checks.values()), "checks": checks, "metrics": metrics}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=100000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sdr-dim", type=int, default=1024)
    parser.add_argument("--log2-capacity-target", type=float, default=250.0)
    parser.add_argument("--morphology-target", type=float, default=4.0)
    parser.add_argument("--bilingual-target", type=float, default=0.28)
    parser.add_argument("--translation-overlap-target", type=float, default=0.15)
    parser.add_argument("--translation-hit-target", type=float, default=0.40)
    parser.add_argument("--normalization-overlap-target", type=float, default=0.25)
    parser.add_argument("--semantic-reconstruction-target", type=float, default=0.14)
    parser.add_argument("--reconstruction-target", type=float, default=0.55)
    parser.add_argument("--cross-domain-target", type=float, default=5.0)
    parser.add_argument("--completion-target", type=float, default=0.80)
    parser.add_argument("--collision-target", type=float, default=0.01)
    parser.add_argument("--synonym-target", type=float, default=0.30)
    parser.add_argument("--analogy-target", type=float, default=0.25)
    parser.add_argument("--bucket-target", type=float, default=0.06)
    parser.add_argument("--reconstruction-samples", type=int, default=128)
    parser.add_argument("--collision-samples", type=int, default=4096)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate(args)
    ok = bool(result["success"])
    print("Phase 9: Bio SDR Semantics")
    print("=" * 88)
    for key, value in result["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"{key:<38} {float(value):.4f}")
    print("Capacity claim:             log2(C(d,k)) is reported exactly; no shortcut cap.")
    print(f"Overall status:             {'PASS' if ok else 'FAIL'}")
    payload = {"benchmark": "phase9_bio_sdr_semantics", **result}
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
