#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
PYTEST_BIN="${PYTEST:-pytest}"

cd "$ROOT_DIR"
unset OPENAI_API_KEY || true

echo "[smoke] help"
"$PYTHON_BIN" rag_system.py --help

echo "[smoke] inspect knowledge base"
"$PYTHON_BIN" rag_system.py inspect-kb

echo "[smoke] build tfidf index"
"$PYTHON_BIN" rag_system.py build --backend tfidf

echo "[smoke] ask offline question"
"$PYTHON_BIN" rag_system.py ask --backend tfidf --llm offline --question "What is self-attention?"

echo "[smoke] run tests"
"$PYTEST_BIN" -q

echo "[smoke] success: offline smoke path completed without OpenAI, ChromaDB, sentence-transformers, or network access."
