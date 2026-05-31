import os
import tempfile

import numpy as np

from lsl import CorticalColumnSequenceMemory, DendriticLayer, LSLCoreModel, NATIVE_AVAILABLE, SimpleSubwordTokenizer


def test_subword_tokenizer_byte_fallback_cache_and_vietnamese():
    with tempfile.TemporaryDirectory() as raw:
        tok = SimpleSubwordTokenizer(
            vocab_size=128,
            vietnamese_normalization=True,
            byte_fallback=True,
            cache_dir=raw,
        )
        tok.build_vocab("Mô hình LSL học tiếng Việt. Bộ nhớ sparse giữ ngữ cảnh.")
        ids = tok.encode("Điện toán LSL giữ dấu tiếng Việt 🚀", max_tokens=256)
        assert ids
        assert tok.unk_rate(ids) == 0.0
        decoded = tok.decode(ids)
        assert "lsl" in decoded
        assert tok.save_cache() is not None
        tok.encode("Điện toán LSL giữ dấu tiếng Việt 🚀", max_tokens=256)
        assert tok.cache_stats()["hit_rate"] > 0.0
        saved = os.path.join(raw, "vi_tokenizer.json")
        tok.save(saved)
        loaded = SimpleSubwordTokenizer.load(saved)
        assert loaded.encode("Điện toán LSL", max_tokens=64)
        assert loaded.unk_rate(loaded.encode("Ký tự mới 🚀", max_tokens=64)) == 0.0


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
    test_subword_tokenizer_byte_fallback_cache_and_vietnamese()
    test_binary_compressed_and_incremental_checkpoint_roundtrip()
    test_native_dendrite_and_cortical_topk_paths()
    print("Tokenizer/checkpoint/native OK")
