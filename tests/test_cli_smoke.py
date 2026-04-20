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
    (tmp_path / "failure_case_report.md").write_text("# Failure Case Report\n", encoding="utf-8")
    return tmp_path


def run_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
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
    assert "Knowledge-base files: 1" in inspect_result.stdout
    assert "FAQ rows: 0" in inspect_result.stdout
    assert "Top 10 chunk IDs: sample::chunk000" in inspect_result.stdout

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
    assert "Retrieval results:" in ask_result.stdout
    assert "ANSWER" in ask_result.stdout
    assert "SOURCES" in ask_result.stdout
    assert "[1]" in ask_result.stdout
    assert "1. source=" in ask_result.stdout
    assert "self-attention" in ask_result.stdout.lower()

    trace_result = run_cli(
        "ask",
        "--backend",
        "tfidf",
        "--llm",
        "offline",
        "--trace-output",
        "--question",
        "What is self-attention?",
        env=env,
    )
    assert trace_result.returncode == 0
    assert "Trace output:" in trace_result.stdout
    assert (temp_root / "results" / "traces" / "latest_retrieval_trace.json").exists()


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
    assert "Retrieval results:" in ask_result.stdout


def test_auto_llm_falls_back_to_offline_without_api_key(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)
    env.pop("OPENAI_API_KEY", None)

    build_result = run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0

    ask_result = run_cli(
        "ask",
        "--backend",
        "auto",
        "--llm",
        "auto",
        "--question",
        "What is self-attention?",
        env=env,
    )
    assert ask_result.returncode == 0
    assert "Resolved llm: offline" in ask_result.stdout
    assert "ANSWER" in ask_result.stdout


def test_ask_does_not_write_trace_without_flag(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)

    build_result = run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0

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
    assert not (temp_root / "results" / "traces" / "latest_retrieval_trace.json").exists()


def test_openai_mode_without_key_errors_clearly(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)
    env.pop("OPENAI_API_KEY", None)

    build_result = run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0

    ask_result = run_cli(
        "ask",
        "--backend",
        "tfidf",
        "--llm",
        "openai",
        "--question",
        "What is self-attention?",
        env=env,
    )
    assert ask_result.returncode != 0
    assert "OPENAI_API_KEY is not set" in ask_result.stderr


def test_demo_offline_creates_markdown_and_default_showcase(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)

    build_result = run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0

    demo_result = run_cli(
        "demo",
        "--backend",
        "tfidf",
        "--llm",
        "offline",
        env=env,
    )
    assert demo_result.returncode == 0
    assert "Demo mode" in demo_result.stdout
    assert "Questions: 5" in demo_result.stdout
    assert "Offline fallback used: false" in demo_result.stdout
    demo_run = temp_root / "results" / "demo_run.md"
    assert demo_run.exists()
    demo_text = demo_run.read_text(encoding="utf-8")
    assert "# Demo Run" in demo_text
    assert "## 1. What is self-attention?" in demo_text
    assert "## 5. What is the capital city of France?" in demo_text


def test_demo_auto_and_legacy_smoke_alias_run_single_question_when_requested(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)
    env.pop("OPENAI_API_KEY", None)

    build_result = run_cli("build", "--backend", "tfidf", env=env)
    assert build_result.returncode == 0

    auto_demo = run_cli(
        "demo",
        "--backend",
        "auto",
        "--llm",
        "auto",
        env=env,
    )
    assert auto_demo.returncode == 0
    assert "Questions: 5" in auto_demo.stdout
    assert "Auto backend fallback: using tfidf" in auto_demo.stdout
    assert "Resolved llm: offline" in auto_demo.stdout

    smoke_demo = run_cli("--smoke-test", "--offline", env=env)
    assert smoke_demo.returncode == 0
    assert "Questions: 1" in smoke_demo.stdout
    demo_text = (temp_root / "results" / "demo_run.md").read_text(encoding="utf-8")
    assert "## 1. What is self-attention?" in demo_text
    assert "## 2." not in demo_text


def test_build_auto_fallback_message_via_main(monkeypatch, capsys) -> None:
    import rag_system

    monkeypatch.setattr(rag_system, "_load_docs_and_chunks", lambda: ("paths", [], []))
    monkeypatch.setattr(
        rag_system,
        "maybe_build_indexes",
        lambda *args, **kwargs: {
            "lexical_index": {"path": "/tmp/tfidf_index.json", "chunk_count": 0},
            "dense_index": {"built": False, "reason": "MiniLM model not cached locally"},
        },
    )

    result = rag_system.main(["build", "--backend", "auto"])
    output = capsys.readouterr().out
    assert result == 0
    assert "Dense index: skipped (MiniLM model not cached locally)" in output
    assert "Auto backend fallback: tfidf will remain the safe local default" in output
