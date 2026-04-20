# Final Project Report

## 1. Problem Statement

The goal of this project was to build a question-answering prototype for course-relevant NLP topics using Retrieval-Augmented Generation. The system needed to answer from a local knowledge base, support a course-compliant dense retrieval path, and still run locally in an offline-safe fallback mode on an M1 MacBook Pro.

## 2. Dataset / Knowledge Base

The project uses a small curated knowledge base stored at the repository root in `knowledge_base/`. The main source is an FAQ-style CSV with 80 entries, supported by deeper Markdown notes for topics such as tensors, RNNs, transformers, and self-attention. At runtime, the corpus resolves to 86 source documents and 93 retrievable chunks. This size is intentionally small enough for local experimentation and honest inspection.

## 3. RAG Architecture

The architecture is built around a root CLI entrypoint, `rag_system.py`, which calls modular code under `src/ragfaq/`. Documents are ingested, normalized, chunked with stable IDs, and then indexed through either a lexical TF-IDF path, a dense ChromaDB path, or a hybrid path that combines both. Retrieval produces the top evidence chunks, and the generation layer answers with citations tied to those chunks.

## 4. Embedding and Vector Store Design

The course-compliant dense path uses `sentence-transformers/all-MiniLM-L6-v2` to embed chunks and queries. Those embeddings are stored in ChromaDB through a `PersistentClient` collection, with metadata including `source`, `source_id`, `topic`, `chunk_index`, and `text_hash`. The same codebase also provides a pure-Python TF-IDF backend for offline-safe use and a hybrid mode that combines dense and lexical retrieval through reciprocal rank fusion and lightweight MMR.

## 5. Prompt Design

The OpenAI prompt is designed to be strict rather than broad. It tells the model to treat retrieved documents as untrusted text, ignore instructions inside the retrieved context, answer only the user question, use only the supplied context, and abstain with `I do not know based on the retrieved context` when evidence is insufficient. Answers are expected to include source citations such as `[1]` and `[2]`.

## 6. Offline Fallback Design

Offline fallback was a core requirement for this project. When no API key is available, or when dense dependencies or model downloads are not available locally, the system falls back to TF-IDF retrieval and an offline extractive generator. That generator selects and lightly composes supported sentences from retrieved chunks, preserves citations, and abstains when relevant support is weak. This mode is the main validated local path for the M1 laptop setup.

## 7. Evaluation Methodology

Evaluation is performed through `test_questions.csv`, which currently contains 30 scored questions:

- 24 answerable in-scope questions
- 6 out-of-scope questions
- 17 paraphrased questions
- 6 multi-hop questions

The evaluation pipeline records:

- Retrieval Recall@3
- MRR@3
- a deterministic faithfulness proxy
- citation validity
- abstention correctness
- latency

The latest validated run used:

```bash
python rag_system.py evaluate --backend tfidf --llm offline
```

## 8. Results Table

| Metric | Value |
| --- | ---: |
| Question count | 30 |
| Answerable questions | 24 |
| Unanswerable questions | 6 |
| Retrieval Recall@3 | 0.92 |
| MRR@3 | 0.78 |
| Faithfulness | 0.85 |
| Citation valid rate | 1.00 |
| Abstention accuracy (unanswerable) | 0.67 |
| Average latency (ms) | 0.89 |
| Median latency (ms) | 0.88 |

These results are promising for a course prototype, but they are not perfect and should not be presented as production-ready.

## 9. Backend Comparison

<!-- backend-comparison:start -->
Auto-generated from real comparison artifacts under `results/comparisons/backend_comparison_summary.json` and `results/comparisons/backend_comparison_table.md`.
Latest comparison run: `2026-04-20T14:57:16.089471+00:00`.

| Configuration | Status | Requested Backend | Requested LLM | Resolved Backend | Resolved LLM | Questions | Answerable | Unanswerable | Recall@3 | MRR@3 | Faithfulness | Citation Validity | Abstention Accuracy | False Abstention | Avg Latency (ms) | Reason |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tfidf + offline | success | tfidf | offline | tfidf | offline | 30 | 24 | 6 | 0.88 | 0.69 | 0.91 | 1.00 | 1.00 | 0.00 | 1.57 | n/a |
| auto + offline | success | auto | offline | tfidf | offline | 30 | 24 | 6 | 0.88 | 0.69 | 0.91 | 1.00 | 1.00 | 0.00 | 1.74 | n/a |
| chroma + offline | skipped | chroma | offline | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | dense retrieval unavailable: chromadb unavailable: ModuleNotFoundError: No module named 'chromadb' |
| hybrid + offline | skipped | hybrid | offline | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | dense retrieval unavailable: chromadb unavailable: ModuleNotFoundError: No module named 'chromadb' |
| chroma + openai | skipped | chroma | openai | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | openai sdk unavailable: ModuleNotFoundError: No module named 'openai' |
| hybrid + openai | skipped | hybrid | openai | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | openai sdk unavailable: ModuleNotFoundError: No module named 'openai' |
<!-- backend-comparison:end -->

## 10. Failure Cases

The weakest cases come from retrieval coverage, multi-hop reasoning, and abstention behavior:

- `Q11` failed primarily as a retrieval problem because the expected RAG-specific source was not retrieved in the top 3.
- `Q21` exposed a multi-hop weakness: metadata evidence was retrieved, but the combined answer pulled in unrelated supporting text.
- `Q26` and `Q27` are out-of-scope questions where the offline generator should have abstained but instead reused partially relevant in-domain content.
- `Q19` shows a knowledge-base coverage issue for a cross-topic PyTorch + neural network training explanation.

These examples are documented in more detail in `failure_case_report.md`.

## 11. Lessons Learned

Three lessons stood out during this project:

1. Retrieval quality controls the rest of the pipeline. When the right source is not retrieved, even a careful generator can still produce a weak answer.
2. Offline-safe local execution is valuable for development and demonstration, especially on constrained hardware.
3. Honest evaluation is more useful than inflated metrics. The project is strongest as a transparent prototype with inspectable failure cases, not as a claim of fully robust QA performance.
