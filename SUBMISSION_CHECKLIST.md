# Project 10 Submission Checklist

This checklist maps each required Project 10 deliverable to the file or command that should satisfy it in the final repository.

| # | Project 10 requirement | File or command that satisfies it | Planned evidence to verify |
| --- | --- | --- | --- |
| 1 | `rag_system.py` | `rag_system.py` | Root CLI entrypoint exists at the repository root |
| 2 | `knowledge_base/` folder | `knowledge_base/` | Root knowledge base folder exists and contains source documents |
| 3 | `test_questions.csv` with scores | `test_questions.csv` | CSV contains 30 questions plus computed score columns |
| 4 | Short report on failure cases | `failure_case_report.md` | Root report summarizes real failure modes and examples |
| 5 | RAG system embeds documents with `sentence-transformers/all-MiniLM-L6-v2` | `python rag_system.py --build-index` | Build/index logs or code path show MiniLM embedding model usage |
| 6 | ChromaDB storage using `collection.add(...)` | `python rag_system.py --build-index` | Stored collection is created through the required Chroma API |
| 7 | Query-time top-3 retrieval using `collection.query(...)` | `python rag_system.py --ask "QUESTION"` | Retrieval code queries Chroma with top 3 results |
| 8 | Prompt says to answer using ONLY the retrieved context | `rag_system.py` | Prompt text explicitly constrains the answer to retrieved context |
| 9 | GPT-4o-mini generation when `OPENAI_API_KEY` is available | `python rag_system.py --ask "QUESTION"` | With `OPENAI_API_KEY` set, generation path uses GPT-4o-mini |
| 10 | Local offline fallback mode when no API key or model download is available | `python rag_system.py --smoke-test --offline` | Offline smoke test runs without requiring `OPENAI_API_KEY` |
| 11 | 30 test questions | `test_questions.csv` | CSV contains exactly 30 evaluation questions |
| 12 | Retrieval Recall@3 and answer faithfulness evaluation | `python rag_system.py --evaluate` | Evaluation run computes metrics and records them honestly |

## Final Verification Pass

- Confirm all root deliverables exist: `rag_system.py`, `knowledge_base/`, `test_questions.csv`, `failure_case_report.md`
- Confirm the command surface matches the project plan exactly
- Confirm `test_questions.csv` includes both retrieval and answer-quality scoring fields
- Confirm offline smoke tests do not depend on an OpenAI API key
- Confirm no evaluation numbers are fabricated or hard-coded

