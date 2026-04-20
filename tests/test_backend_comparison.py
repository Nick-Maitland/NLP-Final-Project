from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.comparison import (
    BACKEND_COMPARISON_END,
    BACKEND_COMPARISON_START,
    sync_backend_comparison_docs,
)
from ragfaq.config import get_paths
from ragfaq.utils import dump_json


def _write_doc_with_markers(path: Path, title: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "## Backend Comparison",
                "",
                BACKEND_COMPARISON_START,
                "placeholder",
                BACKEND_COMPARISON_END,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _prepare_comparison_root(tmp_path: Path) -> Path:
    (tmp_path / "knowledge_base").mkdir()
    (tmp_path / "knowledge_base" / "sample.md").write_text(
        "# Sample Knowledge\n\n"
        "Self-attention lets tokens compare with other tokens in the same sequence.\n\n"
        "Vector store metadata helps trace retrieved chunks back to their source.\n",
        encoding="utf-8",
    )
    with (tmp_path / "test_questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                "question",
                "expected_source_id",
                "expected_topic",
                "answerable",
                "retrieved_source_ids",
                "retrieval_recall_at_3",
                "reciprocal_rank",
                "faithfulness_score",
                "citation_valid",
                "abstention_correct",
                "answer",
                "notes",
            ]
        )
        writer.writerow(
            [
                "Q01",
                "What is self-attention?",
                "sample",
                "attention",
                "true",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "single-hop",
            ]
        )
        writer.writerow(
            [
                "Q02",
                "What is the capital of France?",
                "",
                "out_of_scope",
                "false",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "out_of_scope",
            ]
        )
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "evaluation_summary.json").write_text(
        '{"sentinel": "summary"}\n',
        encoding="utf-8",
    )
    (tmp_path / "results" / "evaluation_report.md").write_text(
        "# Existing Evaluation Report\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "test_questions_scored.csv").write_text(
        "sentinel\n",
        encoding="utf-8",
    )
    (tmp_path / "failure_case_report.md").write_text(
        "# Existing Failure Report\n",
        encoding="utf-8",
    )
    _write_doc_with_markers(tmp_path / "README.md", "README")
    _write_doc_with_markers(tmp_path / "PROJECT_REPORT.md", "Project Report")
    return tmp_path


def _comparison_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(root)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HOME"] = str(root / "fake_home")
    env.pop("OPENAI_API_KEY", None)
    env.pop("HF_HUB_CACHE", None)
    env.pop("TRANSFORMERS_CACHE", None)
    env.pop("HF_HOME", None)
    return env


def _run_comparison(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_backend_comparison.py", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _config_map(summary_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        config["slug"]: config
        for config in summary_payload["configs"]
    }


def test_backend_comparison_offline_only_writes_isolated_outputs(tmp_path: Path) -> None:
    root = _prepare_comparison_root(tmp_path)
    env = _comparison_env(root)
    original_questions = (root / "test_questions.csv").read_text(encoding="utf-8")
    original_summary = (root / "results" / "evaluation_summary.json").read_text(encoding="utf-8")
    original_report = (root / "results" / "evaluation_report.md").read_text(encoding="utf-8")
    original_failure = (root / "failure_case_report.md").read_text(encoding="utf-8")
    original_scored = (root / "results" / "test_questions_scored.csv").read_text(encoding="utf-8")

    result = _run_comparison("--offline-only", env=env)
    assert result.returncode == 0, result.stderr

    summary_path = root / "results" / "comparisons" / "backend_comparison_summary.json"
    table_path = root / "results" / "comparisons" / "backend_comparison_table.md"
    tfidf_csv = root / "results" / "comparisons" / "tfidf_offline_scored.csv"
    auto_csv = root / "results" / "comparisons" / "auto_offline_scored.csv"
    assert summary_path.exists()
    assert table_path.exists()
    assert tfidf_csv.exists()
    assert auto_csv.exists()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    configs = _config_map(summary_payload)
    assert len(configs) == 6
    assert configs["tfidf_offline"]["status"] == "success"
    assert configs["auto_offline"]["status"] == "success"
    assert configs["chroma_offline"]["status"] == "skipped"
    assert configs["hybrid_offline"]["status"] == "skipped"
    assert configs["chroma_openai"]["status"] == "skipped"
    assert configs["hybrid_openai"]["status"] == "skipped"
    assert configs["chroma_openai"]["reason"] == "openai disabled by --offline-only"
    assert configs["hybrid_openai"]["reason"] == "openai disabled by --offline-only"
    assert configs["chroma_offline"]["reason"]
    assert configs["hybrid_offline"]["reason"]

    assert (root / "test_questions.csv").read_text(encoding="utf-8") == original_questions
    assert (root / "results" / "evaluation_summary.json").read_text(encoding="utf-8") == original_summary
    assert (root / "results" / "evaluation_report.md").read_text(encoding="utf-8") == original_report
    assert (root / "failure_case_report.md").read_text(encoding="utf-8") == original_failure
    assert (root / "results" / "test_questions_scored.csv").read_text(encoding="utf-8") == original_scored

    readme_text = (root / "README.md").read_text(encoding="utf-8")
    project_report_text = (root / "PROJECT_REPORT.md").read_text(encoding="utf-8")
    assert "tfidf + offline" in readme_text
    assert "auto + offline" in project_report_text
    assert "results/comparisons/backend_comparison_summary.json" in readme_text


def test_backend_comparison_docs_sync_supports_placeholder_and_generated_table(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_comparison_root(tmp_path)
    monkeypatch.setenv("RAGFAQ_ROOT", str(root))
    paths = get_paths()

    sync_backend_comparison_docs(paths)
    readme_text = paths.readme_path.read_text(encoding="utf-8")
    assert "have not been generated yet" in readme_text

    summary_payload = {
        "generated_at": "2026-04-20T00:00:00+00:00",
        "configs": [
            {
                "label": "tfidf + offline",
                "slug": "tfidf_offline",
                "required": True,
                "status": "success",
                "requested_backend": "tfidf",
                "requested_llm": "offline",
                "resolved_backend": "tfidf",
                "resolved_llm": "offline",
                "questions": 2,
                "answerable_questions": 1,
                "unanswerable_questions": 1,
                "retrieval_recall_at_3": 1.0,
                "mrr_at_3": 1.0,
                "faithfulness": 0.9,
                "citation_validity": 1.0,
                "abstention_accuracy": 1.0,
                "false_abstention_rate": 0.0,
                "average_latency_ms": 1.23,
                "artifact_path": "results/comparisons/tfidf_offline_scored.csv",
                "reason": "",
            }
        ],
    }
    dump_json(paths.backend_comparison_summary_path, summary_payload)
    sync_backend_comparison_docs(paths)
    project_report_text = paths.project_report_path.read_text(encoding="utf-8")
    assert "Latest comparison run:" in project_report_text
    assert "| tfidf + offline | success | tfidf | offline | tfidf | offline | 2 | 1 | 1 | 1.00 |" in project_report_text


def test_openai_rows_skip_without_flag_even_when_key_present(tmp_path: Path) -> None:
    root = _prepare_comparison_root(tmp_path)
    env = _comparison_env(root)
    env["OPENAI_API_KEY"] = "test-key"

    result = _run_comparison(env=env)
    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(
        (root / "results" / "comparisons" / "backend_comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )
    configs = _config_map(summary_payload)
    assert configs["chroma_openai"]["status"] == "skipped"
    assert configs["hybrid_openai"]["status"] == "skipped"
    assert configs["chroma_openai"]["reason"] == "openai disabled"
    assert configs["hybrid_openai"]["reason"] == "openai disabled"


def test_openai_rows_skip_with_flag_when_prerequisites_are_missing(tmp_path: Path) -> None:
    root = _prepare_comparison_root(tmp_path)
    env = _comparison_env(root)

    result = _run_comparison("--include-openai", env=env)
    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(
        (root / "results" / "comparisons" / "backend_comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )
    configs = _config_map(summary_payload)
    if not summary_payload["openai_sdk_available"]:
        assert configs["chroma_openai"]["reason"].startswith("openai sdk unavailable:")
        assert configs["hybrid_openai"]["reason"].startswith("openai sdk unavailable:")
    else:
        assert configs["chroma_openai"]["reason"] == "OPENAI_API_KEY missing"
        assert configs["hybrid_openai"]["reason"] == "OPENAI_API_KEY missing"


def test_openai_rows_skip_for_dense_unavailability_after_key_check(tmp_path: Path) -> None:
    root = _prepare_comparison_root(tmp_path)
    env = _comparison_env(root)
    env["OPENAI_API_KEY"] = "test-key"

    result = _run_comparison("--include-openai", env=env)
    assert result.returncode == 0, result.stderr

    summary_payload = json.loads(
        (root / "results" / "comparisons" / "backend_comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )
    configs = _config_map(summary_payload)
    if not summary_payload["openai_sdk_available"]:
        assert configs["chroma_openai"]["reason"].startswith("openai sdk unavailable:")
        assert configs["hybrid_openai"]["reason"].startswith("openai sdk unavailable:")
    else:
        assert configs["chroma_openai"]["reason"].startswith("dense retrieval unavailable:")
        assert configs["hybrid_openai"]["reason"].startswith("dense retrieval unavailable:")


def test_makefile_compare_targets_map_to_expected_commands() -> None:
    compare_offline = subprocess.run(
        ["make", "-n", "compare-offline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compare_offline.returncode == 0
    assert "scripts/run_backend_comparison.py --offline-only" in compare_offline.stdout
    assert "OPENAI_API_KEY=" in compare_offline.stdout
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in compare_offline.stdout

    compare_full = subprocess.run(
        ["make", "-n", "compare-full"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compare_full.returncode == 0
    assert "scripts/run_backend_comparison.py --include-openai" in compare_full.stdout
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in compare_full.stdout
