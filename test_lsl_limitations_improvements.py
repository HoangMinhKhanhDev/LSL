from lsl import LSLCoreConfig, LSLCoreModel
from lsl.bio import DendriticLayer
from lsl.cortical_column import CorticalColumnSequenceMemory
from lsl.subword_tokenizer import SimpleSubwordTokenizer


def test_subword_encode_max_tokens_is_prefix_bounded():
    text = "the little cat jumped over the little dog. " * 20
    tok = SimpleSubwordTokenizer(vocab_size=128, max_merges=32)
    tok.build_vocab(text)
    full = tok.encode(text)
    limited = tok.encode(text, max_tokens=17)
    assert len(limited) == 17
    assert limited == full[:17]


def test_dendritic_prune_keeps_bounded_prediction_path():
    layer = DendriticLayer(input_dim=128, outputs=16, segment_size=3)
    for idx in range(32):
        layer.observe((idx, idx + 1, idx + 2), output=idx % 16)
    removed = layer.prune_branches(8)
    assert removed == 24
    assert len(layer.branches) == 8
    assert layer.predict((0, 1, 2)) is not None


def test_cortical_column_prune_bounds_sparse_tables():
    memory = CorticalColumnSequenceMemory(vocab_size=64, cells_per_column=16, sparsity=0.125, seed=7)
    for token in list(range(32)) * 4:
        memory.forward(token % 64, learn=True)
    removed = memory.prune_memory(max_segments=12, max_context_keys=10, max_targets_per_context=2)
    assert removed > 0
    assert len(memory.temporal_segments) <= 12
    assert len(memory.context_transitions) <= 10
    assert all(len(targets) <= 2 for targets in memory.context_transitions.values())


def test_bio_native_maintenance_prunes_runtime_sidecars():
    config = LSLCoreConfig(
        vocab_size=128,
        runtime_profile="bio_native",
        bio_maintenance_interval=4,
        bio_dendrite_max_branches=4,
        bio_column_max_segments=8,
        bio_column_max_contexts=8,
        bio_hippocampus_max_fast=4,
        bio_hippocampus_max_slow=4,
        seed=3,
    )
    model = LSLCoreModel(config)
    model.build_tokenizer("a b c d e f g h i j k l m n o p")
    model.fit_tokens(list(range(16)), reset=True)
    diag = model.diagnostics()
    assert diag["bio_native_maintenance_runs"] >= 4.0
    assert diag["bio_native_maintenance_pruned"] > 0.0
    assert len(model.agent.dendrites.branches) <= 4
    assert len(model.agent.columns.temporal_segments) <= 8
    assert len(model.agent.columns.context_transitions) <= 8


if __name__ == "__main__":
    test_subword_encode_max_tokens_is_prefix_bounded()
    test_dendritic_prune_keeps_bounded_prediction_path()
    test_cortical_column_prune_bounds_sparse_tables()
    test_bio_native_maintenance_prunes_runtime_sidecars()
    print("LSL limitation improvements OK")
