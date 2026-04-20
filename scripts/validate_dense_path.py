from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.config import ensure_runtime_directories, get_paths
from ragfaq.dense_validation import run_dense_validation, write_dense_validation_artifacts


def main() -> int:
    paths = ensure_runtime_directories(get_paths())
    summary = run_dense_validation(paths)
    write_dense_validation_artifacts(summary, paths=paths)

    print(f"Dense validation status: {summary['status']}")
    if summary["status"] == "passed":
        print(
            "Dense validation passed: ChromaDB + MiniLM built successfully and "
            "returned exactly three cited chunks."
        )
    elif summary["status"] == "skipped":
        print(f"Dense validation skipped: {summary['reason']}")
    else:
        print(f"Dense validation failed: {summary['reason']}")

    print(f"Summary JSON: {paths.dense_validation_summary_path}")
    print(f"Markdown report: {paths.dense_validation_report_path}")
    return 0 if summary["status"] in {"passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
