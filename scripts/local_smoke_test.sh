#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
PYTEST_BIN="${PYTEST:-}"
PYTHON_SIBLING_PYTEST="$(CDPATH= cd -- "$(dirname -- "$PYTHON_BIN")" && pwd)/pytest"

cd "$ROOT_DIR"
unset OPENAI_API_KEY || true

if [[ -z "$PYTEST_BIN" ]]; then
  if [[ -x "$PYTHON_SIBLING_PYTEST" ]]; then
    PYTEST_BIN="$PYTHON_SIBLING_PYTEST"
  elif command -v pytest >/dev/null 2>&1; then
    PYTEST_BIN="$(command -v pytest)"
  else
    PYTEST_BIN=""
  fi
fi

echo "[smoke] help"
"$PYTHON_BIN" rag_system.py --help

echo "[smoke] inspect knowledge base"
"$PYTHON_BIN" rag_system.py inspect-kb

echo "[smoke] build tfidf index"
"$PYTHON_BIN" rag_system.py --build-index --backend tfidf

echo "[smoke] ask offline question"
"$PYTHON_BIN" rag_system.py --ask "What is self-attention?" --backend tfidf --offline

echo "[smoke] run tests"
if [[ -n "$PYTEST_BIN" ]]; then
  "$PYTEST_BIN" -q
elif "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pytest -q
else
  echo "[smoke] error: pytest is unavailable. Install requirements-lite.txt or run from an environment that already provides pytest." >&2
  exit 1
fi

echo "[smoke] success: offline smoke path completed without OpenAI, ChromaDB, sentence-transformers, or network access."
