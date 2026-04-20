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