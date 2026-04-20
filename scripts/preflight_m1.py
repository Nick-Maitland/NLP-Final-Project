from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragfaq.config import DENSE_MODEL_NAME, get_paths


def _run_probe(import_target: str) -> tuple[bool, str]:
    code = """
import importlib
import json
import sys

sys.path.insert(0, {src_dir!r})
name = {import_target!r}
result = {{"ok": True, "message": ""}}
try:
    importlib.import_module(name)
except Exception as exc:
    result = {{"ok": False, "message": f"{{type(exc).__name__}}: {{exc}}"}}
print(json.dumps(result))
""".format(src_dir=str(SRC_DIR), import_target=import_target)
    process = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
        check=False,
    )
    output = process.stdout.strip().splitlines()
    if not output:
        stderr = process.stderr.strip() or "no output"
        return False, stderr
    try:
        payload = json.loads(output[-1])
    except json.JSONDecodeError:
        return False, output[-1]
    return bool(payload.get("ok")), str(payload.get("message", ""))


def _model_cache_candidates(paths) -> list[Path]:
    candidates: list[Path] = []

    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        candidates.append(Path(env_hub).expanduser())

    env_transformers = os.environ.get("TRANSFORMERS_CACHE")
    if env_transformers:
        candidates.append(Path(env_transformers).expanduser())

    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home:
        hf_home = Path(env_hf_home).expanduser()
        candidates.append(hf_home)
        candidates.append(hf_home / "hub")

    candidates.extend(
        [
            paths.cache_dir / "hf_home",
            paths.cache_dir / "hf_home" / "hub",
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "torch" / "sentence_transformers",
        ]
    )
    unique: list[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _find_model_cache() -> tuple[bool, str]:
    paths = get_paths()
    hub_pattern = "models--sentence-transformers--all-MiniLM-L6-v2"
    st_pattern = "sentence-transformers_all-MiniLM-L6-v2"
    for candidate in _model_cache_candidates(paths):
        if not candidate.exists():
            continue
        if (candidate / hub_pattern).exists():
            return True, str(candidate / hub_pattern)
        if (candidate / st_pattern).exists():
            return True, str(candidate / st_pattern)
    return False, "MiniLM cache not found in known Hugging Face cache locations."


def _offline_fallback_ready(paths) -> tuple[bool, str]:
    checks = [
        (paths.knowledge_base_dir.exists(), "knowledge_base/ is present"),
        ((paths.root_dir / "rag_system.py").exists(), "rag_system.py is present"),
    ]
    for passed, message in checks:
        if not passed:
            return False, message
    package_ok, package_reason = _run_probe("ragfaq")
    if not package_ok:
        return False, f"ragfaq package import failed: {package_reason}"
    return True, "repo-local offline fallback can run"


def _python_status() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11) and (major, minor) < (3, 13)
    return ok, f"{major}.{minor}.{sys.version_info.micro}"


def _safe_mode(
    chroma_ok: bool,
    sentence_ok: bool,
    model_cached: bool,
    offline_ready: bool,
) -> str:
    if chroma_ok and sentence_ok and model_cached:
        return "full"
    if chroma_ok and sentence_ok and not model_cached:
        return "full-with-download"
    if offline_ready:
        return "lite"
    return "blocked"


def main() -> int:
    paths = get_paths()
    python_ok, python_version = _python_status()
    chroma_ok, chroma_reason = _run_probe("chromadb")
    sentence_ok, sentence_reason = _run_probe("sentence_transformers")
    model_cached, model_location = _find_model_cache()
    offline_ready, offline_reason = _offline_fallback_ready(paths)
    safe_mode = _safe_mode(chroma_ok, sentence_ok, model_cached, offline_ready)

    print("RAG FAQ M1 preflight")
    print(f"Repository root: {paths.root_dir}")
    print(f"Python version: {python_version} ({'supported' if python_ok else 'unsupported'})")
    print(f"OPENAI_API_KEY set: {'yes' if os.environ.get('OPENAI_API_KEY') else 'no'}")
    print(f"chromadb import: {'ok' if chroma_ok else 'failed'}")
    if not chroma_ok:
        print(f"  reason: {chroma_reason}")
    print(f"sentence_transformers import: {'ok' if sentence_ok else 'failed'}")
    if not sentence_ok:
        print(f"  reason: {sentence_reason}")
    print(
        f"{DENSE_MODEL_NAME} cached: {'yes' if model_cached else 'no'}"
    )
    print(f"  location: {model_location}")
    print(f"Offline fallback ready: {'yes' if offline_ready else 'no'}")
    print(f"  detail: {offline_reason}")
    print("")
    print(f"SAFE MODE: {safe_mode}")
    if safe_mode == "full":
        print("Recommendation: make setup-full && python scripts/preflight_m1.py")
    elif safe_mode == "full-with-download":
        print("Recommendation: make setup-full")
        print("MiniLM is not cached yet, so the first dense run may need a one-time download.")
    elif safe_mode == "lite":
        print("Recommendation: make setup-lite && make smoke")
        print("The dense stack is not reliable right now, so use tfidf/offline mode locally.")
    else:
        print("Recommendation: fix the Python environment first, then rerun preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
