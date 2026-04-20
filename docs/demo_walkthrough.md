# Optional Streamlit Demo Walkthrough

This walkthrough is for the optional Streamlit UI only. The graded submission path remains the root CLI in `rag_system.py`.

## Launch

Install Streamlit manually only if you want the UI:

```bash
pip install streamlit
streamlit run app.py
```

The app is designed to stay optional:

- Streamlit is not required for `make smoke`
- Streamlit is not required for `make test`
- Streamlit is not required for the CLI workflow

## Recommended Offline Demo Flow

Use the defaults first:

- Backend: `tfidf`
- LLM mode: `offline`
- Question: an in-scope NLP question such as `What is self-attention?`

This path is the safest local demo because it does not depend on OpenAI, ChromaDB, or MiniLM being available at runtime.

## What To Show In A Demo

The UI is organized so you can explain the system from top to bottom:

- Question input: the user prompt being evaluated
- Backend selector: retrieval mode requested for the run
- LLM selector: generation mode requested for the run
- Run summary: backend actually used, LLM mode actually used, latency, and confidence
- Answer tab: grounded answer text, abstention behavior, citation warnings, and fallback notes
- Sources tab: ranked source rows with topic, source path, and retrieval metrics
- Retrieved Context tab: the exact chunks used for answer generation
- Retrieval Trace tab: readable trace summary plus raw JSON for inspection

## How To Demo Fallback Behavior

The app is intentionally forgiving when `auto` is selected:

- `auto` backend warns and stays on TF-IDF when dense dependencies or indexes are unavailable
- `auto` LLM warns and stays on offline mode when the OpenAI key or SDK is unavailable

This makes it easy to demonstrate that the project has:

- a clean offline-safe path
- explicit retrieval traces
- clear runtime diagnostics
- strict behavior for explicit `chroma`, `hybrid`, and `openai` selections

Do not present the Streamlit UI as the required course interface. Present it as an optional portfolio/demo layer over the same local RAG system.
