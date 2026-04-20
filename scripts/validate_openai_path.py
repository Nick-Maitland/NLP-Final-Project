from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.config import ensure_runtime_directories, get_paths
from ragfaq.openai_validation import NON_FAILING_STATUSES, ensure_openai_validation_summary, run_openai_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the optional GPT-4o-mini generation path without calling OpenAI by default.",
    )
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Explicitly allow a small live GPT-4o-mini validation run on three fixed questions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ensure_runtime_directories(get_paths())
    summary = run_openai_validation(run_live=args.run_live, paths=paths)
    ensure_openai_validation_summary(summary, paths=paths)

    print(f"OpenAI validation status: {summary['status']}")
    if summary["status"] == "passed":
        print("OpenAI validation passed: GPT-4o-mini handled the three fixed validation questions.")
    elif summary["status"].startswith("skipped"):
        print(f"OpenAI validation skipped: {summary['reason']}")
    else:
        print(f"OpenAI validation failed: {summary['reason']}")
    print(f"Summary JSON: {paths.openai_validation_summary_path}")
    return 0 if summary["status"] in NON_FAILING_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
