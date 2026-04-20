#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
PYTEST_BIN="${PYTEST:-}"

cd "$ROOT_DIR"
unset OPENAI_API_KEY || true

if [[ -z "$PYTEST_BIN" ]]; then
  if command -v pytest >/dev/null 2>&1; then
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
"$PYTHON_BIN" rag_system.py build --backend tfidf

echo "[smoke] ask offline question"
"$PYTHON_BIN" rag_system.py ask --backend tfidf --llm offline --question "What is self-attention?"

echo "[smoke] run tests"
if [[ -n "$PYTEST_BIN" ]]; then
  "$PYTEST_BIN" -q
else
  "$PYTHON_BIN" -m pytest -q
fi

echo "[smoke] success: offline smoke path completed without OpenAI, ChromaDB, sentence-transformers, or network access."
