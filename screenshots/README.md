# Screenshot Instructions

Add screenshots manually only after you run the optional Streamlit app yourself:

```bash
pip install streamlit
streamlit run app.py
```

No screenshots are generated automatically by this repository, and none should be committed unless they were captured from a real local run.

## Recommended Screenshot Set

Use clear, descriptive filenames such as:

- `01-question-and-controls.png`
- `02-answer-and-summary.png`
- `03-sources-and-context.png`
- `04-retrieval-trace-and-warnings.png`

## What To Capture

- The top of the app with the question input and sidebar controls
- The run summary showing backend used, LLM mode used, latency, and confidence
- The answer panel with citations or abstention behavior
- The sources and retrieved context panels
- A warning state for unavailable dense or OpenAI dependencies
- The retrieval trace panel with the readable summary and raw JSON

## Capture Notes

- Prefer the default offline-safe demo path first: `tfidf` + `offline`
- Capture one in-scope question and, if useful, one out-of-scope abstention example
- If you add screenshots to the repo, keep them in this `screenshots/` directory
