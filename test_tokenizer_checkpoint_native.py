import os
import tempfile

import numpy as np

from lsl import CorticalColumnSequenceMemory, DendriticLayer, LSLCoreModel, NATIVE_AVAILABLE, SimpleSubwordTokenizer
from lsl.text_normalization import looks_vietnamese, normalize_text, repair_utf8_mojibake


def test_text_normalization_and_vietnamese_detection():
    mojibake = "Ti\u1ebfng Vi\u1ec7t".encode("utf-8").decode("latin1")
    assert repair_utf8_mojibake(mojibake) == "Tiếng Việt"
    normalized = normalize_text(
        "\ufeff" + mojibake + "\u200b  \r\n",
        compatibility_normalization=True,
        vietnamese_normalization=True,
    )
    assert "\ufeff" not in normalized
    assert "\u200b" not in normalized
    assert looks_vietnamese("Hôm nay trời đẹp.")
    assert looks_vietnamese(normalized)


def test_subword_tokenizer_byte_fallback_cache_and_vietnamese():
    with tempfile.TemporaryDirectory() as raw:
        tok = SimpleSubwordTokenizer(
            vocab_size=128,
            vietnamese_normalization=True,
            byte_fallback=True,
            cache_dir=raw,
        )
        tok.build_vocab("M\u00f4 h\u00ecnh LSL h\u1ecdc ti\u1ebfng Vi\u1ec7t. B\u1ed9 nh\u1edb sparse gi\u1eef ng\u1eef c\u1ea3nh.")
        ids = tok.encode("\u0110i\u1ec7n to\u00e1n LSL gi\u1eef d\u1ea5u ti\u1ebfng Vi\u1ec7t \U0001f680", max_tokens=256)
        assert ids
        assert tok.unk_rate(ids) == 0.0
        decoded = tok.decode(ids)
        assert "lsl" in decoded
        assert tok.save_cache() is not None
        tok.encode("\u0110i\u1ec7n to\u00e1n LSL gi\u1eef d\u1ea5u ti\u1ebfng Vi\u1ec7t \U0001f680", max_tokens=256)
        assert tok.cache_stats()["hit_rate"] > 0.0
        saved = os.path.join(raw, "vi_tokenizer.json")
        tok.save(saved)
        loaded = SimpleSubwordTokenizer.load(saved)
        assert loaded.encode("\u0110i\u1ec7n to\u00e1n LSL", max_tokens=64)
        assert loaded.unk_rate(loaded.encode("K\u00fd t\u1ef1 m\u1edbi \U0001f680", max_tokens=64)) == 0.0


def test_subword_tokenizer_lru_cache_and_persistent_reload():
    with tempfile.TemporaryDirectory() as raw:
        tok = SimpleSubwordTokenizer(
            vocab_size=128,
            compatibility_normalization=True,
            vietnamese_normalization=True,
            byte_fallback=True,
            cache_dir=raw,
            cache_capacity=2,
        )
        tok.build_vocab("H\u00f4m nay tr\u1eddi \u0111\u1eb9p. LSL h\u1ecdc nhanh v\u00e0 nh\u1edb b\u1ec7n.")
        first = "H\u00f4m nay LSL h\u1ecdc nhanh."
        second = "B\u1ed9 nh\u1edb c\u00f3 d\u1ea5u ti\u1ebfng Vi\u1ec7t."
        third = "Emoji \U0001f680 v\u00e0 k\u00fd t\u1ef1 m\u1edbi."
        ids_first = tok.encode(first, max_tokens=64)
        tok.encode(second, max_tokens=64)
        tok.encode(first, max_tokens=64)
        tok.encode(third, max_tokens=64)
        assert len(tok._encode_text_cache) == 2
        first_key = tok._encode_cache_key(tok._normalize_text(first, lowercase=True), 64)
        second_key = tok._encode_cache_key(tok._normalize_text(second, lowercase=True), 64)
        third_key = tok._encode_cache_key(tok._normalize_text(third, lowercase=True), 64)
        assert first_key in tok._encode_text_cache
        assert second_key not in tok._encode_text_cache
        assert third_key in tok._encode_text_cache
        cache_path = tok.save_cache()
        assert cache_path and os.path.exists(cache_path)
        tok_path = os.path.join(raw, "tokenizer.json")
        tok.save(tok_path)
        loaded = SimpleSubwordTokenizer.load(tok_path)
        assert loaded.enable_cache(raw, load=True) is None
        loaded_stats = loaded.cache_stats()
        assert loaded_stats["encode_entries"] >= 2
        assert loaded_stats["word_entries"] >= 1
        before = loaded_stats["hit_rate"]
        ids_loaded = loaded.encode(first, max_tokens=64)
        assert ids_loaded == ids_first
        assert loaded.cache_stats()["hit_rate"] >= before


def test_binary_compressed_and_incremental_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as raw:
        model = LSLCoreModel(vocab_size=128, runtime_profile="native_fast", seed=3)
        model.train_stream(["a tiny model learns a tiny sequence. a tiny model remembers."], max_tokens=64)
        binary = os.path.join(raw, "model.lslb")
        info = model.save_binary(binary)
        assert info["version"] >= 2
        loaded = LSLCoreModel.load(binary)
        assert loaded.vocab_size == model.vocab_size
        journal = os.path.join(raw, "model.lslj")
        model.save_incremental(journal)
        model.observe("new token path", learn=True)
        model.save_incremental(journal, parent=binary)
        loaded_journal = LSLCoreModel.load(journal)
        assert loaded_journal.seen_tokens == model.seen_tokens
        legacy = os.path.join(raw, "legacy.json")
        migrated = os.path.join(raw, "migrated.lslb")
        model.save(legacy)
        mig = LSLCoreModel.migrate_checkpoint(legacy, migrated)
        assert mig["mode"] == "migrated"
        assert LSLCoreModel.load(migrated).vocab_size == model.vocab_size


def test_native_dendrite_and_cortical_topk_paths():
    layer = DendriticLayer(input_dim=32, outputs=2, segment_size=2)
    layer.add_branch([1, 2], output=1, threshold=1.5)
    layer.add_branch([4, 5], output=0, threshold=1.5)
    assert layer.predict([1, 2]) == 1
    diag = layer.diagnostics()
    if NATIVE_AVAILABLE:
        assert diag["native_predict_success"] >= 1.0

    memory = CorticalColumnSequenceMemory(vocab_size=8, cells_per_column=8, sparsity=0.25)
    scores = np.asarray([0.1, 0.4, 0.2, 0.9, 0.8, 0.0, 0.7, 0.3], dtype=np.float32)
    assert memory.topk_prediction_indices(scores, top_k=3) == [3, 4, 6]
    if NATIVE_AVAILABLE:
        assert memory.metrics()["native_topk_success"] >= 1.0


if __name__ == "__main__":
    test_text_normalization_and_vietnamese_detection()
    test_subword_tokenizer_byte_fallback_cache_and_vietnamese()
    test_subword_tokenizer_lru_cache_and_persistent_reload()
    test_binary_compressed_and_incremental_checkpoint_roundtrip()
    test_native_dendrite_and_cortical_topk_paths()
    print("Tokenizer/checkpoint/native OK")
