from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

from benchmarks.competitive.run_lsl_vs_transformer import run as competitive_run
from lsl.report import write_html_report


def test_competitive_smoke_speed_quality_on_custom_corpus() -> None:
    corpus = (
        "alpha beta gamma alpha beta gamma. "
        "the little cat sat on the mat. "
        "the little dog sat on the rug. "
        "alpha beta gamma delta. "
        "knowledge stays when memory is local. "
    ) * 8
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(corpus)
        args = SimpleNamespace(
            dataset="custom",
            corpus_path=corpus_path,
            train_fraction=0.70,
            max_train_chars=6000,
            max_eval_chars=4000,
            tokenizer_train_chars=4000,
            tokens=512,
            max_train_tokens=512,
            max_eval_tokens=256,
            eval_tokens=64,
            vocab_size=256,
            candidate_cap=64,
            lsl_profile="native_fast",
            trace_memory=False,
            d_model=64,
            context=16,
            context_lengths="8,16,32",
            context_latency_iterations=8,
            transformer_epochs=1,
            lr=0.15,
            prompt_tokens=8,
            generate_tokens=24,
            fact_items=4,
            claim=False,
            quality_ratio_target=1.25,
            latency_speedup_target=1.0,
            loop_rate_target=0.50,
            fact_recall_target=0.50,
            json_output=None,
            results_root=tmpdir,
            seed=42,
        )
        result = competitive_run(args)
        metrics = result["metrics"]
        assert result["success"] is True
        assert metrics["lsl"]["loss"] >= 0.0
        assert metrics["lsl"]["train_tokens_per_second"] > 0.0
        assert metrics["transformer"]["train_tokens_per_second"] > 0.0
        assert metrics["comparison"]["loss_ratio_lsl_over_transformer"] > 0.0
        assert metrics["generation"]["lsl_metrics"]["coherence"] >= 0.0
        assert metrics["generation"]["lsl_metrics"]["loop_rate"] <= 0.50
        assert metrics["generation"]["lsl_metrics"]["unk_rate"] <= 0.50
        assert metrics["online_adaptation"]["works"] is True
        assert metrics["native_core"]["enabled"] >= 1.0
        assert metrics["native_core"]["forward_native_ratio"] >= 0.0


def test_html_report_generation_writes_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        results_root = os.path.join(tmpdir, "results")
        os.makedirs(results_root, exist_ok=True)
        index_path = os.path.join(results_root, "index.jsonl")
        result_path = os.path.join(results_root, "sample.json")
        payload = {
            "benchmark": "sample_benchmark",
            "dataset": "sample_dataset",
            "success": True,
            "metrics": {"loss": 1.23, "tokens_per_second": 456.7},
            "metadata": {
                "benchmark": "sample_benchmark",
                "dataset": "sample_dataset",
                "timestamp": "2026-05-31T00:00:00Z",
                "git_commit": "abc1234",
            },
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"path": result_path, "benchmark": "sample_benchmark", "dataset": "sample_dataset", "timestamp": "2026-05-31T00:00:00Z"}) + "\n")
        output = os.path.join(tmpdir, "report.html")
        rendered = write_html_report(output, results_root=results_root, title="Smoke Report")
        assert os.path.exists(rendered)
        with open(rendered, "r", encoding="utf-8") as f:
            html = f.read()
        assert "Smoke Report" in html
        assert "sample_benchmark" in html
        assert "sample_dataset" in html


if __name__ == "__main__":
    test_competitive_smoke_speed_quality_on_custom_corpus()
    test_html_report_generation_writes_file()
    print("Regression speed/quality OK")
