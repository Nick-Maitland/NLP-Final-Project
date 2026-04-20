from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def prepare_temp_root(tmp_path: Path) -> Path:
    (tmp_path / "knowledge_base").mkdir()
    (tmp_path / "knowledge_base" / "sample.md").write_text(
        "# Sample Knowledge Base\n\nSelf-attention lets each token attend to other tokens "
        "in the same sequence and build contextual representations.",
        encoding="utf-8",
    )
    with (tmp_path / "test_questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                "question",
                "expected_answer",
                "expected_source_id",
                "expected_keywords",
                "retrieved_source_ids",
                "recall_at_3",
                "generated_answer",
                "faithfulness_score",
                "resolved_backend",
                "resolved_llm",
                "notes",
            ]
        )
        writer.writerow(
            [
                "Q01",
                "What is self-attention?",
                "Self-attention lets each token attend to other tokens.",
                "sample",
                "self-attention;token;sequence",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    (tmp_path / "failure_case_report.md").write_text("# Failure Case Report\n", encoding="utf-8")
    return tmp_path


def run_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "rag_system.py", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_works() -> None:
    result = run_cli("--help", env=os.environ.copy())
    assert result.returncode == 0
    assert "inspect-kb" in result.stdout
    assert "--build-index" in result.stdout or "Compatibility aliases" in result.stdout


def test_inspect_build_and_ask_offline(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)

    inspect_result = run_cli("inspect-kb", env=env)
    assert inspect_result.returncode == 0
    assert "Source documents: 1" in inspect_result.stdout

    build_result = run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0
    assert "Lexical index: ready" in build_result.stdout

    ask_result = run_cli(
        "ask",
        "--backend",
        "tfidf",
        "--llm",
        "offline",
        "--question",
        "What is self-attention?",
        env=env,
    )
    assert ask_result.returncode == 0
    assert "Resolved backend: tfidf" in ask_result.stdout
    assert "Resolved llm: offline" in ask_result.stdout
    assert "self-attention" in ask_result.stdout.lower()


def test_auto_backend_falls_back_to_tfidf_with_message(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)

    build_result = run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0

    ask_result = run_cli(
        "ask",
        "--backend",
        "auto",
        "--llm",
        "offline",
        "--question",
        "What is self-attention?",
        env=env,
    )
    assert ask_result.returncode == 0
    assert "Auto backend fallback: using tfidf" in ask_result.stdout
    assert "Resolved backend: tfidf" in ask_result.stdout

