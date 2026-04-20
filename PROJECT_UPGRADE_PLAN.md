# Project 10 Upgrade Plan

## Current File Structure

Current tracked repository structure:

```text
NLP-Final-Project/
+-- LICENSE
```

Current local state:

- The repository is a valid Git checkout on `main`.
- The tracked project content currently contains only `LICENSE`.
- None of the Project 10 implementation files or deliverables exist yet.

## Gaps Versus Project 10 Requirements

| Project 10 requirement | Current repo status | Gap to close |
| --- | --- | --- |
| `rag_system.py` at repo root | Missing | Add root entrypoint and keep it stable for grading |
| `knowledge_base/` at repo root | Missing | Add root knowledge base folder with source documents |
| `test_questions.csv` with scores | Missing | Add 30-question evaluation CSV with computed metrics |
| Short report on failure cases | Missing | Add `failure_case_report.md` at repo root |
| Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings | Missing | Implement embedding pipeline with this exact model |
| Use ChromaDB storage via `collection.add(...)` | Missing | Build and persist a Chroma collection through the required API |
| Query-time top-3 retrieval via `collection.query(...)` | Missing | Retrieve exactly top 3 chunks per query |
| Prompt says to answer using only retrieved context | Missing | Add explicit constrained-answer prompt |
| GPT-4o-mini generation when `OPENAI_API_KEY` is available | Missing | Add OpenAI-backed generation path |
| Local offline fallback when no API key or model download is available | Missing | Add deterministic local fallback mode that still runs on the laptop |
| 30 test questions | Missing | Add 30 representative FAQ questions to `test_questions.csv` |
| Retrieval Recall@3 and answer faithfulness evaluation | Missing | Add evaluation pipeline and record scores honestly |

## Proposed Improved Architecture

### Root-level deliverables that must exist

- `rag_system.py`
- `knowledge_base/`
- `test_questions.csv`
- `failure_case_report.md`

### Lightweight internal structure

The final project should stay course-spec-friendly at the root while using small helper modules for maintainability:

```text
NLP-Final-Project/
+-- rag_system.py
+-- knowledge_base/
+-- test_questions.csv
+-- failure_case_report.md
+-- requirements.txt
+-- README.md
+-- rag_components/
    +-- ingest.py
    +-- embed.py
    +-- store.py
    +-- retrieve.py
    +-- generate.py
    +-- evaluate.py
    +-- fallback.py
```

### Responsibilities

- `rag_system.py`: single grading-friendly CLI entrypoint for indexing, asking questions, evaluation, and offline smoke tests
- `knowledge_base/`: small local FAQ/source files that can be embedded on a laptop without heavy setup
- `rag_components/ingest.py`: load and chunk documents from `knowledge_base/`
- `rag_components/embed.py`: load `sentence-transformers/all-MiniLM-L6-v2` when available and expose a consistent embedding interface
- `rag_components/store.py`: initialize and persist ChromaDB collections and write documents using `collection.add(...)`
- `rag_components/retrieve.py`: perform `collection.query(...)` with `n_results=3`
- `rag_components/generate.py`: construct the strict context-only prompt and use GPT-4o-mini when `OPENAI_API_KEY` is present
- `rag_components/fallback.py`: provide deterministic offline answer generation when API access or model download is unavailable
- `rag_components/evaluate.py`: compute Recall@3 and answer faithfulness and write results to `test_questions.csv`

## What Will Remain Course-Spec-Compliant

The final implementation must preserve these exact course-facing behaviors:

- Keep `rag_system.py` at the repository root.
- Keep `knowledge_base/` at the repository root.
- Keep `test_questions.csv` at the repository root.
- Keep `failure_case_report.md` at the repository root.
- Use `sentence-transformers/all-MiniLM-L6-v2` for the main embedding path.
- Use ChromaDB storage with `collection.add(...)`.
- Use query-time top-3 retrieval with `collection.query(...)`.
- Use a prompt that explicitly instructs the system to answer using only the retrieved context.
- Use GPT-4o-mini for generation when `OPENAI_API_KEY` is available.
- Support a local offline fallback mode when the API key is absent or the model cannot be downloaded.
- Include 30 test questions.
- Report Retrieval Recall@3 and answer faithfulness with honest computed scores.

## What Will Be Extra / Resume-Impressive

These additions improve polish without breaking the course spec or making the project heavy:

- Small maintainable modules behind a single root CLI
- Persisted local Chroma database so repeated runs do not rebuild unnecessarily
- Source chunk IDs shown with answers for transparency
- Clear CLI modes for indexing, interactive QA, evaluation, and smoke testing
- Deterministic offline fallback that still demonstrates retrieval and constrained answering
- Honest metrics and failure analysis rather than hand-written or inflated evaluation claims
- Clean README usage examples tailored to local execution on Apple Silicon

## Local Offline Strategy For M1 Mac

Target machine: M1 MacBook Pro with 16 GB unified memory.

Constraints:

- No heavy local LLMs
- No FAISS GPU
- No Docker
- No dependencies that are likely to be painful on Apple Silicon
- Offline smoke tests must succeed without `OPENAI_API_KEY`

Planned strategy:

- Default to a lightweight Python stack with ChromaDB and `sentence-transformers`
- Use `sentence-transformers/all-MiniLM-L6-v2` as the primary embedding model when it is already available or can be loaded locally
- If the MiniLM download is unavailable, fall back to a deterministic local retrieval path for smoke tests rather than failing the whole project
- Keep the offline fallback generator extractive and rule-based, using only the top retrieved chunks and never inventing unsupported claims
- Avoid any runtime path that attempts to pull multi-GB local models
- Keep the knowledge base small enough that indexing and querying remain practical on CPU on an M1 laptop
- Make offline smoke tests use preexisting local assets only

## Exact Commands The Final Project Should Support

```bash
python rag_system.py --build-index
python rag_system.py --ask "QUESTION"
python rag_system.py --ask "QUESTION" --offline
python rag_system.py --evaluate
python rag_system.py --smoke-test --offline
```

Expected command behavior:

- `--build-index`: ingest `knowledge_base/`, embed documents, and store them in ChromaDB
- `--ask "QUESTION"`: run standard RAG, using GPT-4o-mini when the API key is available
- `--ask "QUESTION" --offline`: run retrieval plus local fallback generation without requiring network APIs
- `--evaluate`: run the 30-question evaluation, compute Recall@3 and faithfulness, and update output artifacts honestly
- `--smoke-test --offline`: verify that the project works locally in offline mode without requiring an API key or large model download

## Implementation Order

1. Add root deliverable files and a lightweight dependency list.
2. Add a small knowledge base suitable for FAQ retrieval.
3. Implement indexing with MiniLM embeddings and Chroma `collection.add(...)`.
4. Implement top-3 retrieval with `collection.query(...)`.
5. Add the strict context-only prompting path for GPT-4o-mini.
6. Add deterministic offline fallback behavior.
7. Add 30 test questions and evaluation metrics.
8. Write the failure-case report from real evaluation results.

