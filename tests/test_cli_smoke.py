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
    (tmp_path / "PROJECT_REPORT.md").write_text(
        "\n".join(
            [
                "# Project Report",
                "",
                "<!-- dense-validation:start -->",
                "placeholder",
                "<!-- dense-validation:end -->",
                "",
            ]
        ),
        encoding="utf-8",
    )
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


def run_dense_validation_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    command_env = env.copy()
    command_env.setdefault("HF_HUB_OFFLINE", "1")
    command_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    command_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return subprocess.run(
        [sys.executable, "scripts/validate_dense_path.py"],
        cwd=REPO_ROOT,
        env=command_env,
        capture_output=True,
        text=True,
        check=False,
    )


def prepare_shadow_missing_dense_modules(tmp_path: Path) -> Path:
    shadow_dir = tmp_path / "shadow_modules"
    shadow_dir.mkdir()
    (shadow_dir / "chromadb.py").write_text(
        'raise ImportError("shadowed chromadb import failure")\n',
        encoding="utf-8",
    )
    (shadow_dir / "sentence_transformers.py").write_text(
        'raise ImportError("shadowed sentence_transformers import failure")\n',
        encoding="utf-8",
    )
    return shadow_dir


def test_help_works() -> None:
    result = run_cli("--help", env=os.environ.copy())
    assert result.returncode == 0
    assert "inspect-kb" in result.stdout
    assert "--build-index" in result.stdout or "Compatibility aliases" in result.stdout


def test_repo_inspect_kb_reports_faq_count_and_clean_chunks() -> None:
    env = os.environ.copy()
    inspect_result = run_cli("inspect-kb", env=env)
    assert inspect_result.returncode == 0
    assert "FAQ rows: 102" in inspect_result.stdout
    assert "Too-short chunks: none" in inspect_result.stdout
    assert "Duplicate-like chunks skipped: 0" in inspect_result.stdout


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


def test_explicit_chroma_commands_fail_clearly_without_silent_fallback(tmp_path: Path) -> None:
    temp_root = prepare_temp_root(tmp_path)
    shadow_dir = prepare_shadow_missing_dense_modules(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)
    env["PYTHONPATH"] = str(shadow_dir)

    build_result = run_cli("build", "--backend", "chroma", "--rebuild", env=env)
    assert build_result.returncode != 0
    assert "Dense build was requested but unavailable." in build_result.stderr
    assert "Install the full dense stack from requirements.txt" in build_result.stderr
    assert "all-MiniLM-L6-v2" in build_result.stderr
    assert "Auto backend fallback" not in build_result.stdout

    ask_result = run_cli(
        "ask",
        "--backend",
        "chroma",
        "--llm",
        "offline",
        "--question",
        "What is self-attention?",
        env=env,
    )
    assert ask_result.returncode != 0
    assert "Dense retrieval requested but unavailable." in ask_result.stderr
    assert "Build the dense index with `python rag_system.py build --backend chroma --rebuild`" in ask_result.stderr
    assert "Auto backend fallback" not in ask_result.stdout

    evaluate_result = run_cli("evaluate", "--backend", "chroma", "--llm", "offline", env=env)
    assert evaluate_result.returncode != 0
    assert "Dense build was requested but unavailable." in evaluate_result.stderr
    assert "Auto backend fallback" not in evaluate_result.stdout


def test_validate_dense_path_script_prints_clear_skip_message_when_dense_deps_missing(
    tmp_path: Path,
) -> None:
    temp_root = prepare_temp_root(tmp_path)
    shadow_dir = prepare_shadow_missing_dense_modules(tmp_path)
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(temp_root)
    env["PYTHONPATH"] = str(shadow_dir)

    result = run_dense_validation_script(env=env)
    assert result.returncode == 0
    assert "Dense validation status: skipped" in result.stdout
    assert "Dense validation skipped:" in result.stdout
    assert (temp_root / "results" / "dense_validation_summary.json").exists()
    assert (temp_root / "results" / "dense_validation_report.md").exists()


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
