"""Smoke tests for the unified LSL core model."""
import os
import tempfile

from lsl import LSLCoreModel, NATIVE_AVAILABLE


def main():
    text = (
        "alpha opens gate one. beta opens gate two. alpha opens gate one. "
        "the local model remembers sparse transitions and answers facts. "
    ) * 4
    model = LSLCoreModel(vocab_size=512, seed=7)
    metrics = model.train_stream([text], max_tokens=400)
    assert metrics["tokens"] > 20
    eval_metrics = model.evaluate_text(text)
    assert eval_metrics["loss"] < 5.0, eval_metrics
    generated = model.generate("alpha opens", max_new_tokens=12)
    assert generated
    diag = model.diagnostics()
    assert diag["seen_tokens"] >= metrics["tokens"]
    if NATIVE_AVAILABLE:
        assert diag["native_core_enabled"] == 1.0, diag
        assert diag["native_core_forward_native_ratio"] == 1.0, diag
        assert diag["native_core_update_native_ratio"] == 1.0, diag
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "core.pkl")
        model.save(path)
        loaded = LSLCoreModel.load(path)
        assert loaded.evaluate_text(text)["loss"] < 5.0
        before = loaded.diagnostics()["seen_tokens"]
        loaded.train_stream(["delta epsilon zeta"], max_tokens=24)
        assert loaded.diagnostics()["seen_tokens"] > before

    bio = LSLCoreModel(vocab_size=512, seed=11, runtime_profile="bio_native")
    bio.train_stream([text], max_tokens=160)
    bio_diag = bio.diagnostics()
    assert bio_diag["runtime_profile"] == "bio_native"
    assert bio_diag["bio_native_pc_steps"] > 0.0, bio_diag
    assert bio_diag["bio_native_sdr_steps"] > 0.0, bio_diag
    assert bio_diag["bio_native_column_steps"] > 0.0, bio_diag
    assert bio_diag["bio_native_hippocampus_writes"] > 0.0, bio_diag
    assert bio_diag["bio_native_neuromod_steps"] > 0.0, bio_diag
    assert bio_diag["bio_native_dendrite_writes"] > 0.0, bio_diag
    assert bio.generate("alpha opens", max_new_tokens=8)

    continual = LSLCoreModel(vocab_size=512, seed=13, runtime_profile="continual")
    continual.train_stream([text], max_tokens=160)
    continual_diag = continual.diagnostics()
    assert continual_diag["runtime_profile"] == "bio_native"
    assert continual_diag["bio_native_pc_steps"] > 0.0, continual_diag
    assert continual_diag["bio_native_column_steps"] > 0.0, continual_diag
    print("LSLCoreModel OK")


if __name__ == "__main__":
    main()
