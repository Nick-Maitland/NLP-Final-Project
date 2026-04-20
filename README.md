# RAG FAQ Project

This repository implements Durham College NLP Project 10 as a RAG-based FAQ answering system.

## Layout

- `rag_system.py`: root CLI required by the course
- `src/ragfaq/`: internal package with ingestion, retrieval, generation, and evaluation code
- `knowledge_base/`: repo-local curated source documents
- `test_questions.csv`: 30 evaluation questions plus scores populated by the evaluation command
- `failure_case_report.md`: failure analysis generated from real evaluation runs

## Lite Setup (offline-safe)

```bash
make setup-lite
```

Lite setup is the recommended first run on an M1 MacBook Pro. It installs only the
dependencies needed for TF-IDF retrieval, offline extractive answers, evaluation, and tests.
It does not require OpenAI, ChromaDB, sentence-transformers, or model downloads.

## Full Setup (dense + OpenAI capable)

```bash
make setup-full
```

Full setup installs the complete Project 10 stack, including ChromaDB and
`sentence-transformers/all-MiniLM-L6-v2` support. The first dense run may require a one-time
MiniLM model download if the model is not already cached.

## M1 Preflight

```bash
python scripts/preflight_m1.py
```

This checks whether the current environment is best suited for:

- `SAFE MODE: lite`
- `SAFE MODE: full`
- `SAFE MODE: full-with-download`
- `SAFE MODE: blocked`

## Make Targets

```bash
make setup-lite
make setup-full
make smoke
make evaluate-offline
make test
make clean
make package
```

`make smoke` and `make test` are designed to work in lite mode without network access, OpenAI,
ChromaDB, or sentence-transformers.

## Primary Commands

```bash
python rag_system.py build
python rag_system.py ask --question "What is self-attention?"
python rag_system.py evaluate
python rag_system.py inspect-kb
python rag_system.py demo
```

## Key Modes

- `--backend chroma`: ChromaDB plus `sentence-transformers/all-MiniLM-L6-v2`
- `--backend tfidf`: pure-Python lexical retrieval that works offline
- `--backend hybrid`: combines dense and lexical retrieval when both are ready
- `--backend auto`: prefers hybrid, then chroma, then tfidf
- `--llm openai`: GPT-4o-mini when `OPENAI_API_KEY` is available
- `--llm offline`: extractive grounded answer generation with no network calls
- `--llm auto`: prefers OpenAI and falls back to offline mode

## Compatibility Aliases

The earlier planning docs remain valid because these legacy forms still work:

```bash
python rag_system.py --build-index
python rag_system.py --ask "What is self-attention?" --offline
python rag_system.py --evaluate
python rag_system.py --smoke-test --offline
```

## Offline Smoke Path

These commands are expected to work without an OpenAI API key, without ChromaDB, and without a
sentence-transformers model download:

```bash
python rag_system.py inspect-kb
python rag_system.py build --backend tfidf
python rag_system.py ask --backend tfidf --llm offline --question "What is self-attention?"
```

The local smoke script runs the same offline-safe path:

```bash
scripts/local_smoke_test.sh
```
