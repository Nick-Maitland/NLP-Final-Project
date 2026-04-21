# System Card

## 1. System name

Prototype RAG FAQ Answering System for Durham College NLP Project 10.

## 2. Intended use

This system is a prototype for course-support question answering over a small local NLP knowledge base. It is intended for class demonstrations, local experimentation, and evaluation of grounded answers that cite retrieved course material.

## 3. Out-of-scope use

This project is not production-ready and should not be used as an authoritative academic, legal, medical, financial, or open-domain assistant. It is also out of scope for answering beyond the local knowledge base or for high-stakes decisions.

## 4. Knowledge base contents

The knowledge base is a small curated local corpus under `knowledge_base/`. It includes an FAQ CSV plus supporting Markdown notes on tensors, PyTorch basics, RNNs, sequence modeling, transformers, self-attention, and related course topics. The current runtime corpus described in the project report resolves to 86 source documents and 93 retrievable chunks.

## 5. Retrieval methods

The validated local retrieval path is pure-Python TF-IDF. The repository also implements an optional dense path using `sentence-transformers/all-MiniLM-L6-v2` with ChromaDB and an optional hybrid mode that combines lexical and dense retrieval with reciprocal rank fusion and lightweight MMR. Dense and hybrid paths require optional dependencies, local model/runtime availability, and separate validation in the current environment.

## 6. Generation methods

The safest validated generation path is an offline extractive generator that selects and lightly composes supported sentences from retrieved chunks and keeps citations. An optional OpenAI path uses GPT-4o-mini with a prompt that instructs the model to answer only from retrieved context and abstain when support is insufficient. OpenAI use requires an API key, the SDK, and separate validation.

## 7. Offline fallback behavior

When the OpenAI path, dense dependencies, or local MiniLM runtime are unavailable, the project falls back to `tfidf` retrieval with `offline` generation. This is the strongest validated path in the current repo and the safest default for local use on an M1 Mac. Offline extractive generation is limited and can still produce weak or over-composed answers when retrieval is only loosely related.

## 8. Evaluation benchmark

The benchmark is the canonical 30-question evaluation set in `evaluation_questions.csv`. Evaluation writes the course-facing scored output to `test_questions.csv` and a richer scored artifact to `results/test_questions_scored.csv`. The latest validated benchmark described in this repository is the offline-safe `tfidf` + `offline` run.

## 9. Metrics reported

The current validated offline benchmark reports 30 questions total, with 24 answerable and 6 out-of-scope. Reported metrics are Retrieval Recall@3 of 0.88, MRR@3 of 0.69, faithfulness of 0.91, citation validity of 1.00, abstention accuracy on unanswerable questions of 1.00, average latency of 1.51 ms, and median latency of 1.40 ms.

## 10. Known limitations

This prototype has narrow coverage because it answers only from the local knowledge base. Offline generation is extractive and limited, not a full generative reasoning system. Dense retrieval and OpenAI generation are optional paths that may be implemented but unavailable or unvalidated in some environments. Answers should be treated as course-support answers grounded in the knowledge base, not authoritative facts beyond it.

## 11. Failure modes

Known failure modes include retrieval misses that lead to incomplete or weak answers, multi-hop questions that need evidence spread across multiple sources, and imperfect abstention on loosely related out-of-scope questions. The dense path may fail to run when ChromaDB, sentence-transformers, or a compatible local MiniLM setup is missing. The OpenAI path may be skipped when `OPENAI_API_KEY` or the OpenAI SDK is absent.

## 12. Privacy/security notes

The offline path keeps processing local to the machine and works without sending questions to a remote model. If the optional OpenAI path is used, retrieved context and the user question may be sent to the API for answer generation. The project does not claim hardened security controls, formal privacy review, or production data governance.

## 13. Environmental and hardware notes for M1 Mac

The project is designed to remain usable on an M1 MacBook Pro with 16 GB unified memory by keeping the offline-safe path lightweight. Lite setup avoids large model downloads and does not require ChromaDB or OpenAI. Dense retrieval can be more fragile on local environments because it depends on optional packages, compatible tokenizer and transformer versions, and local MiniLM availability.

## 14. Recommended safe use

Use this system as a transparent course prototype: prefer `tfidf` + `offline` for local runs, inspect citations and retrieved context, and treat answers as grounded summaries of the course materials rather than final authority. When confidence is low or the answer seems incomplete, check the cited sources directly instead of relying on the generated response alone.
