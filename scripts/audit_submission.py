from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REQUIRED_FILES = [
    "rag_system.py",
    "evaluation_questions.csv",
    "test_questions.csv",
    "failure_case_report.md",
    "README.md",
    "PROJECT_REPORT.md",
]

REQUIRED_EVALUATION_QUESTION_COLUMNS = [
    "question_id",
    "question",
    "expected_source_id",
    "expected_topic",
    "answerable",
    "question_type",
    "difficulty",
    "notes",
]

REQUIRED_TEST_QUESTION_COLUMNS = [
    "retrieval_recall_at_3",
    "faithfulness_score",
]

CODE_PATTERNS: dict[str, tuple[str, ...]] = {
    "Chroma code path exists": (
        "chromadb.PersistentClient",
        "collection.add(",
        "collection.query(",
    ),
    "MiniLM embedding path exists": (
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
    "GPT-4o-mini path exists": (
        "gpt-4o-mini",
    ),
    "Offline fallback path exists": (
        "OfflineExtractiveGenerator",
        "--llm offline",
        'OFFLINE = "offline"',
    ),
}


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    details: str


SmokeTestRunner = Callable[[Path], tuple[bool, str]]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_or_name(path: Path, root_dir: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return path.name


def _check_file_exists(root_dir: Path, relative_path: str) -> AuditCheck:
    path = root_dir / relative_path
    return AuditCheck(
        name=f"{relative_path} exists",
        passed=path.is_file(),
        details=_relative_or_name(path, root_dir),
    )


def _check_knowledge_base(root_dir: Path) -> AuditCheck:
    knowledge_base_dir = root_dir / "knowledge_base"
    if not knowledge_base_dir.is_dir():
        return AuditCheck(
            name="knowledge_base/ exists and is non-empty",
            passed=False,
            details="knowledge_base/ directory is missing",
        )

    file_count = sum(1 for path in knowledge_base_dir.rglob("*") if path.is_file())
    return AuditCheck(
        name="knowledge_base/ exists and is non-empty",
        passed=file_count > 0,
        details=f"{file_count} files found",
    )


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def _check_evaluation_question_schema(root_dir: Path) -> AuditCheck:
    csv_path = root_dir / "evaluation_questions.csv"
    if not csv_path.is_file():
        return AuditCheck(
            name="evaluation_questions.csv matches canonical benchmark schema",
            passed=False,
            details="evaluation_questions.csv is missing",
        )
    _, fieldnames = _read_csv_rows(csv_path)
    passed = fieldnames == REQUIRED_EVALUATION_QUESTION_COLUMNS
    return AuditCheck(
        name="evaluation_questions.csv matches canonical benchmark schema",
        passed=passed,
        details=(
            "canonical field order confirmed"
            if passed
            else f"available columns: {', '.join(fieldnames)}"
        ),
    )


def _check_csv_row_count(root_dir: Path, relative_path: str, *, exact_count: int) -> AuditCheck:
    csv_path = root_dir / relative_path
    check_name = f"{relative_path} has exactly {exact_count} rows"
    if not csv_path.is_file():
        return AuditCheck(
            name=check_name,
            passed=False,
            details=f"{relative_path} is missing",
        )
    rows, _ = _read_csv_rows(csv_path)
    return AuditCheck(
        name=check_name,
        passed=len(rows) == exact_count,
        details=f"{len(rows)} rows found",
    )


def _check_test_question_count(root_dir: Path) -> AuditCheck:
    csv_path = root_dir / "test_questions.csv"
    if not csv_path.is_file():
        return AuditCheck(
            name="test_questions.csv has exactly 30 rows",
            passed=False,
            details="test_questions.csv is missing",
        )
    rows, _ = _read_csv_rows(csv_path)
    return AuditCheck(
        name="test_questions.csv has exactly 30 rows",
        passed=len(rows) == 30,
        details=f"{len(rows)} rows found",
    )


def _check_test_question_columns(root_dir: Path) -> list[AuditCheck]:
    csv_path = root_dir / "test_questions.csv"
    if not csv_path.is_file():
        return [
            AuditCheck(
                name=f"test_questions.csv contains {column}",
                passed=False,
                details="test_questions.csv is missing",
            )
            for column in REQUIRED_TEST_QUESTION_COLUMNS
        ]
    _, fieldnames = _read_csv_rows(csv_path)
    return [
        AuditCheck(
            name=f"test_questions.csv contains {column}",
            passed=column in fieldnames,
            details=f"available columns: {', '.join(fieldnames)}",
        )
        for column in REQUIRED_TEST_QUESTION_COLUMNS
    ]


def _scan_repository_code(root_dir: Path) -> str:
    search_paths = [root_dir / "rag_system.py", root_dir / "src"]
    segments: list[str] = []
    for path in search_paths:
        if path.is_file():
            segments.append(path.read_text(encoding="utf-8"))
            continue
        for file_path in sorted(path.rglob("*.py")):
            segments.append(file_path.read_text(encoding="utf-8"))
    return "\n".join(segments)


def _check_code_patterns(root_dir: Path) -> list[AuditCheck]:
    combined_text = _scan_repository_code(root_dir)
    checks: list[AuditCheck] = []
    for name, patterns in CODE_PATTERNS.items():
        missing = [pattern for pattern in patterns if pattern not in combined_text]
        checks.append(
            AuditCheck(
                name=name,
                passed=not missing,
                details="all required code markers found"
                if not missing
                else f"missing markers: {', '.join(missing)}",
            )
        )
    return checks


def resolve_pytest_binary() -> str | None:
    existing = os.environ.get("PYTEST")
    if existing:
        return existing

    sibling = Path(sys.executable).with_name("pytest")
    if sibling.exists():
        return str(sibling)

    which_pytest = shutil.which("pytest")
    if which_pytest:
        return which_pytest

    return None


def default_smoke_test_runner(root_dir: Path) -> tuple[bool, str]:
    pytest_binary = resolve_pytest_binary()
    if not pytest_binary:
        return False, "could not resolve pytest executable from PYTEST env, sibling binary, or PATH"

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHON"] = sys.executable
    env["PYTEST"] = pytest_binary

    command = ["bash", "scripts/local_smoke_test.sh"]
    completed = subprocess.run(
        command,
        cwd=root_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True, f"{' '.join(command)} passed"

    output_parts = [segment.strip() for segment in (completed.stdout, completed.stderr) if segment.strip()]
    tail = " | ".join(output_parts)[-400:]
    return False, f"{' '.join(command)} failed with exit code {completed.returncode}: {tail}"


def run_audit(
    root_dir: Path | None = None,
    *,
    smoke_test_runner: SmokeTestRunner | None = None,
) -> list[AuditCheck]:
    repo_root = root_dir or get_repo_root()
    checks: list[AuditCheck] = []

    for relative_path in REQUIRED_FILES:
        checks.append(_check_file_exists(repo_root, relative_path))

    checks.append(_check_knowledge_base(repo_root))
    checks.append(_check_evaluation_question_schema(repo_root))
    checks.append(_check_csv_row_count(repo_root, "evaluation_questions.csv", exact_count=30))
    checks.append(_check_test_question_count(repo_root))
    checks.extend(_check_test_question_columns(repo_root))
    checks.extend(_check_code_patterns(repo_root))

    smoke_runner = smoke_test_runner or default_smoke_test_runner
    smoke_passed, smoke_details = smoke_runner(repo_root)
    checks.append(
        AuditCheck(
            name="local smoke test command passes",
            passed=smoke_passed,
            details=smoke_details,
        )
    )
    return checks


def print_audit_report(checks: list[AuditCheck]) -> None:
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.details}")

    passed = sum(1 for check in checks if check.passed)
    failed = len(checks) - passed
    print()
    print("Summary")
    print(f"Total checks: {len(checks)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Submission checklist: {'PASS' if failed == 0 else 'FAIL'}")


def main() -> int:
    checks = run_audit()
    print_audit_report(checks)
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
