from __future__ import annotations

import zipfile
from pathlib import Path

try:
    from audit_submission import AuditCheck, SmokeTestRunner, print_audit_report, run_audit
except ImportError:  # pragma: no cover - fallback for package-style imports
    from .audit_submission import AuditCheck, SmokeTestRunner, print_audit_report, run_audit


SUBMISSION_ARCHIVE_NAME = "NLP-Final-Project-submission.zip"
INCLUDE_PATHS = [
    "rag_system.py",
    "app.py",
    "src",
    "knowledge_base",
    "evaluation_questions.csv",
    "test_questions.csv",
    "failure_case_report.md",
    "PROJECT_REPORT.md",
    "README.md",
    "SYSTEM_CARD.md",
    "SUBMISSION_CHECKLIST.md",
    "requirements.txt",
    "requirements-lite.txt",
    "Makefile",
    "docs/demo_walkthrough.md",
    "screenshots/README.md",
    ".github/workflows/offline-ci.yml",
    "scripts",
    "tests",
    "results/evaluation_summary.json",
    "results/evaluation_report.md",
    "results/dense_validation_summary.json",
    "results/dense_validation_report.md",
    "results/test_questions_scored.csv",
]
EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".ragfaq",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "chroma_db",
    "dist",
    "results/traces",
}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_hidden_or_cache_part(part: str) -> bool:
    lowered = part.lower()
    return part.startswith(".") or "cache" in lowered or lowered in {"hub", "models"}


def _should_exclude(relative_path: Path) -> bool:
    relative_text = relative_path.as_posix()
    if any(relative_text == excluded or relative_text.startswith(f"{excluded}/") for excluded in EXCLUDED_DIR_NAMES):
        return True
    if relative_path.suffix in EXCLUDED_FILE_SUFFIXES:
        return True
    return any(_is_hidden_or_cache_part(part) for part in relative_path.parts[:-1])


def collect_submission_files(root_dir: Path) -> list[Path]:
    collected: list[Path] = []
    for include in INCLUDE_PATHS:
        include_path = root_dir / include
        if not include_path.exists():
            raise FileNotFoundError(f"required submission path is missing: {include}")

        if include_path.is_file():
            collected.append(include_path)
            continue

        for file_path in sorted(path for path in include_path.rglob("*") if path.is_file()):
            relative_path = file_path.relative_to(root_dir)
            if _should_exclude(relative_path):
                continue
            collected.append(file_path)

    # Preserve order while removing duplicates.
    unique_files: list[Path] = []
    seen: set[Path] = set()
    for file_path in collected:
        relative_path = file_path.relative_to(root_dir)
        if relative_path in seen:
            continue
        seen.add(relative_path)
        unique_files.append(file_path)
    return unique_files


def create_submission_zip(root_dir: Path, output_path: Path) -> int:
    files_to_package = collect_submission_files(root_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files_to_package:
            archive.write(file_path, arcname=file_path.relative_to(root_dir))

    return len(files_to_package)


def build_submission_package(
    root_dir: Path | None = None,
    *,
    smoke_test_runner: SmokeTestRunner | None = None,
) -> tuple[list[AuditCheck], Path | None, int]:
    repo_root = root_dir or get_repo_root()
    checks = run_audit(repo_root, smoke_test_runner=smoke_test_runner)
    if not all(check.passed for check in checks):
        return checks, None, 0

    output_path = repo_root / "dist" / SUBMISSION_ARCHIVE_NAME
    file_count = create_submission_zip(repo_root, output_path)
    return checks, output_path, file_count


def main() -> int:
    repo_root = get_repo_root()
    checks, output_path, file_count = build_submission_package(repo_root)
    print_audit_report(checks)
    if not all(check.passed for check in checks):
        return 1

    assert output_path is not None
    print(f"Packaged files: {file_count}")
    print(f"Submission zip: {output_path.resolve()}")
    print("Submission checklist: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
