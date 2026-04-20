# AGENTS.md

## Repo Purpose

This repository is for Durham College NLP Project 10: a RAG-based FAQ answering system that must remain easy to run locally on an M1 MacBook Pro with 16 GB unified memory.

## Non-Negotiable Repository Layout

- Preserve `rag_system.py` at the root.
- Keep `knowledge_base/` at the root.
- Keep `test_questions.csv` at the root.
- Keep `failure_case_report.md` at the root.

Do not move these deliverables into `src/`, `app/`, notebooks, or nested package directories. Helper modules are fine, but the grading-facing files must stay at the repository root.

## Course-Spec Constraints

- Use `sentence-transformers/all-MiniLM-L6-v2` for the main embedding path.
- Use ChromaDB storage with `collection.add(...)`.
- Use query-time top-3 retrieval with `collection.query(...)`.
- Use a prompt that says the answer must use only the retrieved context.
- Use GPT-4o-mini when `OPENAI_API_KEY` is available.
- Maintain a real local offline fallback mode when the API key is missing or model download is unavailable.
- Keep exactly 30 evaluation questions unless the assignment spec changes.
- Report Retrieval Recall@3 and answer faithfulness from actual evaluation runs.

## Testing And Execution Rules

- Never require an OpenAI API key for offline smoke tests.
- Never download large models during tests.
- Do not fake evaluation results.
- Run local tests before claiming completion.
- Keep offline smoke tests deterministic and lightweight.
- Prefer commands that work from the repository root with `python rag_system.py ...`.

## Implementation Preferences

- Prefer clear, small, maintainable Python modules.
- Keep the primary user-facing CLI in `rag_system.py`.
- Use lightweight dependencies that work well on Apple Silicon.
- Do not introduce heavy local LLM runtimes, Docker-only flows, FAISS GPU setups, or anything likely to be painful on an M1 laptop.
- Prefer a simple extractive or rule-based fallback mode over complicated offline generation stacks.
- Keep the knowledge base compact enough for local indexing and querying on CPU.

## Editing Guidance For Future Sessions

- Do not refactor away the root-script workflow just to make the project look more "production-like".
- If helper modules are added, keep them small and easy to trace from `rag_system.py`.
- Preserve the exact command surface planned for this repository:

```bash
python rag_system.py --build-index
python rag_system.py --ask "QUESTION"
python rag_system.py --ask "QUESTION" --offline
python rag_system.py --evaluate
python rag_system.py --smoke-test --offline
```

- When changing evaluation behavior, update `test_questions.csv` and `failure_case_report.md` together.
- When touching retrieval or prompting logic, verify it still satisfies the course requirements before adding optional improvements.
- If a result cannot be computed honestly, leave it blank or mark it clearly as not yet run instead of inventing values.

