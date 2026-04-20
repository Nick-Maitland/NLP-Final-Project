from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_openai_validation_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    command_env = env.copy()
    command_env.setdefault("HF_HUB_OFFLINE", "1")
    command_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    command_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return subprocess.run(
        [sys.executable, "scripts/validate_openai_path.py"],
        cwd=REPO_ROOT,
        env=command_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_openai_path_script_skips_cleanly_without_api_key(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["RAGFAQ_ROOT"] = str(tmp_path)
    env.pop("OPENAI_API_KEY", None)

    result = _run_openai_validation_script(env=env)
    assert result.returncode == 0
    assert "OpenAI validation status: skipped_no_api_key" in result.stdout
    assert "OpenAI validation skipped:" in result.stdout

    summary_path = tmp_path / "results" / "openai_validation_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "skipped_no_api_key"
    assert summary["run_live"] is False


def test_makefile_validate_openai_target_defaults_to_skip_only_command() -> None:
    result = subprocess.run(
        ["make", "-n", "validate-openai"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "scripts/validate_openai_path.py" in result.stdout
    assert "--run-live" not in result.stdout


def test_makefile_validate_openai_target_allows_explicit_live_flag() -> None:
    result = subprocess.run(
        ["make", "-n", "validate-openai", "RUN_LIVE=1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "scripts/validate_openai_path.py --run-live" in result.stdout
