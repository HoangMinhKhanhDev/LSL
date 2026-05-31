import numpy as np

from lsl import DendriticLayer, SimpleSubwordTokenizer
from lsl.memory import SparseKeyValueMemory
from lsl import sparse_native


def test_native_tokenize_matches_python():
    text = "Mô hình LSL học tiếng Việt, and native hot paths."
    native = sparse_native.simple_tokenize(text)
    assert native[:4] == ["mô", "hình", "lsl", "học"]
    tok = SimpleSubwordTokenizer(vocab_size=128, byte_fallback=True, vietnamese_normalization=True)
    tok.build_vocab(text)
    ids = tok.encode(text, max_tokens=64)
    assert ids
    assert tok.unk_rate(ids) == 0.0


def test_native_best_signature_match_matches_python():
    query = np.asarray([1, 2, 3, 9], dtype=np.intp)
    signatures = np.asarray(
        [
            [1, 2, 5, -1],
            [1, 3, 4, -1],
            [2, 3, 9, -1],
        ],
        dtype=np.intp,
    )
    lengths = np.asarray([3, 3, 3], dtype=np.intp)
    values = np.asarray([10, 20, 30], dtype=np.intp)
    native = sparse_native.best_signature_match(query, signatures, lengths, values)
    assert int(native["best_value"]) == 30
    assert int(native["best_score"]) == 3


def test_sparse_memory_native_lookup_and_fallback_agree():
    memory = SparseKeyValueMemory(capacity=64, sdr_dim=256, sparsity=0.02, candidate_cap=16)
    for i in range(32):
        memory.add(i, i * 3, vocab_size=512)
    native_value = memory.lookup(7, vocab_size=512, allow_direct_lookup=False, prefer_native_scoring=True)
    fallback_value = memory.lookup(7, vocab_size=512, allow_direct_lookup=False, prefer_native_scoring=False)
    assert native_value == fallback_value


def test_native_dendrite_predict_and_batch_kernel():
    layer = DendriticLayer(input_dim=32, outputs=2, segment_size=2)
    layer.add_branch([1, 2], output=1, threshold=1.5)
    layer.add_branch([4, 5], output=0, threshold=1.5)
    assert layer.predict([1, 2], prefer_native=True) == 1
    assert layer.predict([1, 2], prefer_native=False) == 1

    slow = np.random.default_rng(0).random((8, 16), dtype=np.float32)
    live = np.random.default_rng(1).random((8, 16), dtype=np.float32)
    fatigue = np.zeros((8, 16), dtype=np.float32)
    active = np.asarray([[1, 2, 3], [2, 4, 6]], dtype=np.intp)
    values = np.ones((2, 3), dtype=np.float32)
    lengths = np.asarray([3, 2], dtype=np.intp)
    post, stats = sparse_native.forward_active_batch(slow, live, fatigue, active, values, lengths)
    assert post.shape == (2, 8)
    assert int(stats["batch"]) == 2


if __name__ == "__main__":
    test_native_tokenize_matches_python()
    test_native_best_signature_match_matches_python()
    test_sparse_memory_native_lookup_and_fallback_agree()
    test_native_dendrite_predict_and_batch_kernel()
    print("Native kernel tests OK")
