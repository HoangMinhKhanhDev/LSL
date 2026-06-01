from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from lsl.curriculum import apply_curriculum_preset, build_curriculum_plan, evaluate_scale_readiness, run_curriculum


def test_curriculum_smoke_three_stage_run() -> None:
    corpus = (
        "alpha beta gamma. the cat sits. the dog runs. "
        "grammar helps the model keep order. "
    ) * 24
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "adapt.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(corpus)

        args = SimpleNamespace(
            bootstrap_dataset="tinystories",
            bootstrap_tokens=256,
            bootstrap_profile="native_fast",
            broaden_datasets="vietnamese_small,dialogue_small",
            broaden_tokens="128,128",
            broaden_profile="continual",
            adapt_dataset="custom",
            adapt_corpus_path=corpus_path,
            adapt_tokens=128,
            adapt_profile="continual",
            adapt_repeat_small=True,
            load_checkpoint=None,
            checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
            final_checkpoint=os.path.join(tmpdir, "final.json"),
            tokenizer_train_chars=2048,
            eval_tokens=64,
            generate_tokens=16,
            chars_per_token=4,
            train_fraction=0.70,
            vocab_size=256,
            candidate_cap=64,
            ood_items=1,
            ood_eval_tokens=64,
            seed=42,
            results_root=os.path.join(tmpdir, "results"),
            json_output=os.path.join(tmpdir, "curriculum.json"),
            smoke=False,
        )

        payload = run_curriculum(args)

        assert payload["success"] is True
        assert payload["summary"]["stage_count"] == 3
        assert len(payload["stages"]) == 3
        assert os.path.exists(payload["final_checkpoint"])
        assert os.path.exists(payload["result_path"])
        assert payload["retention"]
        assert payload["final_evaluation"]
        for stage in payload["stages"]:
            assert os.path.exists(stage["checkpoint_path"])
            assert stage["train_runs"]
            assert stage["metrics"]["stage_tokens"] > 0.0
            assert stage["metrics"]["stage_eval_loss"] > 0.0


def test_scale_ready_preset_splits_stage_three_into_substages() -> None:
    args = SimpleNamespace(
        preset="scale_ready",
        bootstrap_dataset="tinystories",
        bootstrap_tokens=100000,
        broaden_datasets="wikitext2,vietnamese_small",
        broaden_tokens="25000,25000",
        adapt_dataset="wikitext2",
        adapt_tokens=1000000,
        adapt_token_splits=None,
        adapt_profile="continual",
        adapt_repeat_small=False,
        ood_items=4,
        ood_eval_tokens=256,
    )
    apply_curriculum_preset(args)
    plan = build_curriculum_plan(args)
    assert [stage.name for stage in plan] == [
        "bootstrap_grammar",
        "broaden_language",
        "continual_adaptation_1",
        "continual_adaptation_2",
        "continual_adaptation_3",
        "continual_adaptation_4",
    ]


def test_scale_readiness_gate_tracks_retention_grammar_throughput_and_ood() -> None:
    stage_payloads = [
        {
            "name": "bootstrap_grammar",
            "retention": {"mean_loss_ratio": 1.0},
            "metrics": {"stage_grammar_coherence": 0.91, "stage_train_tps": 120000.0},
        },
        {
            "name": "broaden_language",
            "retention": {"mean_loss_ratio": 0.999},
            "metrics": {"stage_grammar_coherence": 0.89, "stage_train_tps": 8000.0},
        },
        {
            "name": "continual_adaptation_1",
            "retention": {"mean_loss_ratio": 0.996},
            "metrics": {"stage_grammar_coherence": 0.86, "stage_train_tps": 4200.0},
        },
    ]
    passing = evaluate_scale_readiness(stage_payloads, {"summary": {"mean_loss": 4.30, "mean_accuracy": 0.14}})
    assert passing["pass"] is True
    assert passing["observed"]["grammar_coherence"] == 0.86
    failing = evaluate_scale_readiness(stage_payloads, {"summary": {"mean_loss": 4.50, "mean_accuracy": 0.10}})
    assert failing["pass"] is False
    assert failing["failed_checks"]


if __name__ == "__main__":
    test_curriculum_smoke_three_stage_run()
    test_scale_ready_preset_splits_stage_three_into_substages()
    test_scale_readiness_gate_tracks_retention_grammar_throughput_and_ood()
    print("LSL curriculum OK")
