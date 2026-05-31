from lsl import DatasetConfig, DatasetLoader, LSLCoreModel
from lsl.results import run_metadata, write_result


def test_dataset_loader_named_corpora_and_splits():
    loader = DatasetLoader()
    available = set(loader.list_available_datasets())
    assert "vietnamese_small" in available
    assert "dialogue_small" in available
    assert "wikitext2" in available
    splits = loader.load_text_splits("vietnamese_small", max_train_chars=2000, max_eval_chars=500)
    assert "LSL" in splits.train
    assert splits.language == "vi"
    stats = loader.compute_stats(DatasetConfig("dialogue_small"))
    assert stats.total_lines > 5
    assert stats.total_words > 20


def test_dataset_loader_token_budget_repeat():
    loader = DatasetLoader()
    model = LSLCoreModel(vocab_size=128, runtime_profile="native_fast", seed=7)
    text = loader.load_text(DatasetConfig("dialogue_small", max_chars=3000))
    model.build_tokenizer(text)
    tokens = loader.load_tokens(
        DatasetConfig("dialogue_small", max_chars=3000, repeat=True),
        model.tokenizer,
        max_tokens=128,
    )
    assert len(tokens) == 128


def test_result_metadata_writer(tmp_path):
    output = tmp_path / "result.json"
    payload = {"success": True, "metrics": {"x": 1.0}}
    path = write_result(
        payload,
        benchmark="phase1_test",
        dataset="dialogue_small",
        seed=3,
        config={"tiny": True},
        output_path=str(output),
        results_root=str(tmp_path),
    )
    assert path == str(output)
    text = output.read_text(encoding="utf-8")
    assert "phase1_test" in text
    assert "git_commit" in text
    meta = run_metadata("phase1_test", "dialogue_small", seed=3)
    assert meta["seed"] == 3


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_dataset_loader_named_corpora_and_splits()
    test_dataset_loader_token_budget_repeat()
    with tempfile.TemporaryDirectory() as raw:
        test_result_metadata_writer(Path(raw))
    print("Phase 1 foundation OK")
