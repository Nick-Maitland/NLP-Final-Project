# RAG FAQ Project

This repository implements Durham College NLP Project 10 as a RAG-based FAQ answering system.

## Layout

- `rag_system.py`: root CLI required by the course
- `src/ragfaq/`: internal package with ingestion, retrieval, generation, and evaluation code
- `knowledge_base/`: repo-local curated source documents
- `test_questions.csv`: 30 evaluation questions plus scores populated by the evaluation command
- `failure_case_report.md`: failure analysis generated from real evaluation runs

## Install

```bash
python3 -m pip install -r requirements.txt
```

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

These commands are expected to work without an OpenAI API key and without a sentence-transformers model download:

```bash
python rag_system.py inspect-kb
python rag_system.py build --backend tfidf
python rag_system.py ask --backend tfidf --llm offline --question "What is self-attention?"
```

