from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_eval_root(tmp_path: Path) -> Path:
    (tmp_path / "knowledge_base").mkdir()
    (tmp_path / "knowledge_base" / "sample.md").write_text(
        "# Sample Knowledge\n\n"
        "Self-attention lets tokens compare with other tokens in the same sequence.\n\n"
        "Vector store metadata helps trace retrieved chunks back to their source.",
        encoding="utf-8",
    )
    with (tmp_path / "evaluation_questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                "question",
                "expected_source_id",
                "expected_topic",
                "answerable",
                "question_type",
                "difficulty",
                "notes",
            ]
        )
        writer.writerow(
            [
                "Q01",
                "How does self-attention help with context?",
                "sample",
                "attention",
                "true",
                "single-hop",
                "intro",
                "paraphrase",
            ]
        )
        writer.writerow(
            [
                "Q02",
                "What is the capital of France?",
                "",
                "out_of_scope",
                "false",
                "out_of_scope",
                "advanced",
                "geography",
            ]
        )
    (tmp_path / "failure_case_report.md").write_text("# Failure Case Report\n", encoding="utf-8")
    return tmp_path


def _run_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    command_env = env.copy()
    command_env.setdefault("HF_HUB_OFFLINE", "1")
    command_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    command_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return subprocess.run(
        [sys.executable, "rag_system.py", *args],
        cwd=REPO_ROOT,
        env=command_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_evaluate_writes_root_and_results_outputs(tmp_path: Path) -> None:
    temp_root = _prepare_eval_root(tmp_path)
    original_benchmark = (temp_root / "evaluation_questions.csv").read_text(encoding="utf-8")
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)
    env.pop("OPENAI_API_KEY", None)

    build_result = _run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0

    evaluate_result = _run_cli("evaluate", "--backend", "tfidf", "--llm", "offline", env=env)
    assert evaluate_result.returncode == 0
    assert "Scored CSV:" in evaluate_result.stdout
    assert "Summary JSON:" in evaluate_result.stdout

    root_rows = list(csv.DictReader((temp_root / "test_questions.csv").open()))
    assert len(root_rows) == 2
    assert root_rows[0]["question_type"] == "single-hop"
    assert root_rows[0]["difficulty"] == "intro"
    assert root_rows[0]["retrieval_recall_at_3"] != ""
    assert root_rows[1]["retrieval_recall_at_3"] == ""
    assert root_rows[1]["abstention_correct"] in {"true", "false"}

    scored_rows = list(csv.DictReader((temp_root / "results" / "test_questions_scored.csv").open()))
    assert scored_rows[0]["question_type"] == "single-hop"
    assert scored_rows[0]["difficulty"] == "intro"
    assert "latency_ms" in scored_rows[0]
    assert "citation_warnings" in scored_rows[0]
    assert "retrieved_chunk_ids" in scored_rows[0]
    assert (
        (temp_root / "evaluation_questions.csv").read_text(encoding="utf-8") == original_benchmark
    )

    summary = json.loads((temp_root / "results" / "evaluation_summary.json").read_text())
    assert summary["question_count"] == 2
    assert "retrieval_recall_at_3_answerable" in summary
    assert "false_abstention_rate_answerable" in summary
    assert "per_topic" in summary

    report = (temp_root / "results" / "evaluation_report.md").read_text(encoding="utf-8")
    assert "## Summary Metrics" in report
    assert "## Weak Examples" in report

    failure_report = (temp_root / "failure_case_report.md").read_text(encoding="utf-8")
    assert "## Concrete Weak Examples" in failure_report
    assert "Failure type:" in failure_report
