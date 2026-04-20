from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_submission
import package_submission


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_evaluation_questions_csv(path: Path) -> None:
    fieldnames = [
        "question_id",
        "question",
        "expected_source_id",
        "expected_topic",
        "answerable",
        "question_type",
        "difficulty",
        "notes",
    ]

    rows = []
    for index in range(30):
        answerable = index < 24
        rows.append(
            {
                "question_id": f"Q{index + 1}",
                "question": f"Question {index + 1}?",
                "expected_source_id": "faq_dummy_001" if answerable else "",
                "expected_topic": "demo" if answerable else "out_of_scope",
                "answerable": "true" if answerable else "false",
                "question_type": "single-hop" if answerable else "out_of_scope",
                "difficulty": "intro" if answerable else "advanced",
                "notes": "" if answerable else "geography",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_test_questions_csv(path: Path, include_faithfulness: bool = True) -> None:
    fieldnames = [
        "question_id",
        "question",
        "expected_source_id",
        "expected_topic",
        "answerable",
        "question_type",
        "difficulty",
        "retrieved_source_ids",
        "retrieval_recall_at_3",
        "reciprocal_rank",
        "citation_valid",
        "abstention_correct",
        "answer",
        "notes",
    ]
    if include_faithfulness:
        fieldnames.insert(fieldnames.index("citation_valid"), "faithfulness_score")

    rows = []
    for index in range(30):
        row = {
            "question_id": f"Q{index + 1}",
            "question": f"Question {index + 1}?",
            "expected_source_id": "faq_dummy_001",
            "expected_topic": "demo",
            "answerable": "true",
            "question_type": "single-hop",
            "difficulty": "intro",
            "retrieved_source_ids": "faq_dummy_001",
            "retrieval_recall_at_3": "1.0",
            "reciprocal_rank": "1.0",
            "citation_valid": "true",
            "abstention_correct": "true",
            "answer": "Supported answer [1]",
            "notes": "",
        }
        if include_faithfulness:
            row["faithfulness_score"] = "1.0"
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_minimal_submission_repo(root: Path, *, include_faithfulness: bool = True) -> None:
    _write_text(
        root / "rag_system.py",
        '"""stub"""\n'
        'HELP = "--llm offline"\n',
    )
    _write_text(root / "failure_case_report.md", "# Failure Case Report\n")
    _write_text(root / "README.md", "# README\n")
    _write_text(root / "PROJECT_REPORT.md", "# Project Report\n")
    _write_text(root / "requirements.txt", "pytest\n")
    _write_text(root / "requirements-lite.txt", "pytest\n")
    _write_text(root / "Makefile", "package:\n\t@echo package\n")
    _write_text(root / "knowledge_base" / "faqs.csv", "source_id,question,answer,topic,difficulty\nfaq_dummy_001,Q,A,demo,intro\n")
    _write_text(
        root / "src" / "ragfaq" / "stub.py",
        "\n".join(
            [
                'MODEL = "sentence-transformers/all-MiniLM-L6-v2"',
                'OPENAI_MODEL = "gpt-4o-mini"',
                "chromadb.PersistentClient",
                "collection.add(",
                "collection.query(",
                "class OfflineExtractiveGenerator: ...",
                'OFFLINE = "offline"',
            ]
        ),
    )
    _write_text(root / "scripts" / "local_smoke_test.sh", "#!/usr/bin/env bash\nexit 0\n")
    _write_text(root / "scripts" / "check_environment.py", "print('ok')\n")
    _write_text(root / "tests" / "test_dummy.py", "def test_dummy():\n    assert True\n")
    _write_text(root / "results" / "evaluation_summary.json", "{}\n")
    _write_text(root / "results" / "evaluation_report.md", "# Report\n")
    _write_text(root / "results" / "dense_validation_summary.json", "{}\n")
    _write_text(root / "results" / "dense_validation_report.md", "# Dense Validation Report\n")
    _write_text(root / "results" / "test_questions_scored.csv", "question_id\nQ1\n")
    _write_evaluation_questions_csv(root / "evaluation_questions.csv")
    _write_test_questions_csv(root / "test_questions.csv", include_faithfulness=include_faithfulness)


def test_submission_audit_passes_on_current_repo_with_mocked_smoke() -> None:
    checks = audit_submission.run_audit(
        ROOT_DIR,
        smoke_test_runner=lambda _: (True, "mock smoke passed"),
    )
    assert all(check.passed for check in checks)


def test_submission_audit_fails_when_faithfulness_column_missing(tmp_path: Path) -> None:
    _create_minimal_submission_repo(tmp_path, include_faithfulness=False)

    checks = audit_submission.run_audit(
        tmp_path,
        smoke_test_runner=lambda _: (True, "mock smoke passed"),
    )

    faithfulness_check = next(
        check for check in checks if check.name == "test_questions.csv contains faithfulness_score"
    )
    assert not faithfulness_check.passed


def test_submission_package_creates_whitelisted_zip_and_excludes_caches(tmp_path: Path) -> None:
    _create_minimal_submission_repo(tmp_path)
    _write_text(tmp_path / ".venv" / "bin" / "python", "ignore\n")
    _write_text(tmp_path / ".git" / "HEAD", "ignore\n")
    _write_text(tmp_path / "src" / "__pycache__" / "stub.pyc", "ignore\n")
    _write_text(tmp_path / "chroma_db" / "data.sqlite3", "ignore\n")
    _write_text(tmp_path / "results" / "traces" / "latest_retrieval_trace.json", "{}\n")

    checks, output_path, file_count = package_submission.build_submission_package(
        tmp_path,
        smoke_test_runner=lambda _: (True, "mock smoke passed"),
    )

    assert all(check.passed for check in checks)
    assert output_path == tmp_path / "dist" / package_submission.SUBMISSION_ARCHIVE_NAME
    assert output_path is not None and output_path.exists()
    assert file_count > 0

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())

    assert "rag_system.py" in names
    assert "src/ragfaq/stub.py" in names
    assert "knowledge_base/faqs.csv" in names
    assert "evaluation_questions.csv" in names
    assert "results/evaluation_summary.json" in names
    assert "results/dense_validation_summary.json" in names
    assert "results/dense_validation_report.md" in names
    assert ".venv/bin/python" not in names
    assert ".git/HEAD" not in names
    assert "src/__pycache__/stub.pyc" not in names
    assert "chroma_db/data.sqlite3" not in names
    assert "results/traces/latest_retrieval_trace.json" not in names
