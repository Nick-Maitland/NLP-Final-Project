# Project 10 Submission Checklist

This checklist records the current repository evidence for the final Project 10 submission. It is written as a proof-of-presence document, not as a future plan. The project is a prototype, and the main validated path on the M1 laptop is the offline-safe TF-IDF + extractive generation configuration.

## Required Submission Evidence

| Requirement | Current evidence |
| --- | --- |
| Root `rag_system.py` entrypoint | Present at `rag_system.py` and used by the documented commands `python rag_system.py build ...`, `ask ...`, `evaluate ...`, `inspect-kb`, and `demo ...` |
| Root `knowledge_base/` folder | Present at `knowledge_base/` with `faqs.csv` plus supporting notes under `knowledge_base/docs/` |
| Scored `test_questions.csv` | Present at `test_questions.csv` with 30 scored rows and the columns `retrieval_recall_at_3`, `reciprocal_rank`, `faithfulness_score`, `citation_valid`, `abstention_correct`, and `answer` |
| Failure-case report | Present at `failure_case_report.md` with concrete weak examples from the latest evaluation run |
| ChromaDB implementation | Dense vector-store code is present under `src/ragfaq/vector_store.py` and uses `chromadb.PersistentClient`, `collection.add(...)`, and `collection.query(...)` |
| `sentence-transformers/all-MiniLM-L6-v2` embedding path | Dense embedding configuration is present in `src/ragfaq/config.py` and implemented in `src/ragfaq/embeddings.py` through `SentenceTransformerEmbeddingProvider` |
| GPT-4o-mini generation path | OpenAI-backed generation is present in `src/ragfaq/generation.py` through `OpenAIGenerator`, which targets `gpt-4o-mini` when `OPENAI_API_KEY` is available |
| Offline fallback path | Offline-safe retrieval and answer generation are present through `--backend tfidf`, `--llm offline`, and `--backend auto --llm auto` fallback behavior |
| 30-question evaluation set | `test_questions.csv` contains 30 scored questions, which is the enforced course-sized evaluation set used by this prototype |
| Retrieval Recall@3 evaluation | The current aggregate Recall@3 result is recorded in `results/evaluation_summary.json` as `0.92` |
| Faithfulness evaluation | The current aggregate faithfulness result is recorded in `results/evaluation_summary.json` as `0.85` |

## Course-Compliant Commands

These commands match the current CLI surface and the repository documentation:

```bash
python rag_system.py inspect-kb
python rag_system.py build --backend tfidf
python rag_system.py ask --backend tfidf --llm offline --question "What is self-attention?"
python rag_system.py evaluate --backend tfidf --llm offline
python rag_system.py demo --backend tfidf --llm offline
```

When the optional dense stack and cached MiniLM model are available locally, the course-compliant dense path is:

```bash
python rag_system.py build --backend chroma --rebuild
python rag_system.py ask --backend chroma --llm openai --question "What is self-attention?"
```

## Verified Artifacts

- `results/evaluation_summary.json`
- `results/evaluation_report.md`
- `results/test_questions_scored.csv`
- `results/demo_run.md`

## Notes On Validation Scope

- The dense ChromaDB path and GPT-4o-mini path are implemented in code and documented in the repository.
- The main repeated local validation path on the M1 laptop is the offline-safe configuration because it does not require `OPENAI_API_KEY`, `chromadb`, `sentence-transformers`, or model downloads.
- The repository therefore satisfies the class structure while still providing a stronger prototype path when the optional dense/OpenAI stack is available.
