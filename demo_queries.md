# Demo Queries

These are the five curated questions used by `python rag_system.py demo` when no `--question` is supplied.

1. `What is self-attention?`
   - Rationale: Shows a direct in-scope transformer concept.
   - Expected behavior: `answerable`
2. `Why can transformers train more in parallel than RNNs?`
   - Rationale: Demonstrates a classic architecture comparison.
   - Expected behavior: `answerable`
3. `Why is retrieval helpful before answer generation in a RAG system?`
   - Rationale: Shows the project's retrieval-augmented workflow.
   - Expected behavior: `answerable`
4. `Why does metadata matter after a vector store retrieves a chunk?`
   - Rationale: Highlights source tracing and citation support.
   - Expected behavior: `answerable`
5. `What is the capital city of France?`
   - Rationale: Demonstrates abstention on an out-of-scope question.
   - Expected behavior: `out_of_scope`
