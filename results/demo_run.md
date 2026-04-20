# Demo Run

- Timestamp: 2026-04-20T07:14:11.220710+00:00
- Requested backend: `tfidf`
- Requested llm: `offline`
- Question count: 5
- Resolved backends used: tfidf
- Resolved llm modes used: offline
- Any offline fallback used: false
- Average latency (ms): 1.03

## 1. What is self-attention?

- Rationale: Shows the core transformer mechanism with a direct in-scope definition.
- Expected behavior category: `answerable`
- Resolved backend: `tfidf`
- Resolved llm: `offline`
- Latency (ms): 1.48
- Offline fallback used: false

### Answer

Self attention lets tokens in the same sequence compare with one another so each token representation can incorporate broader context. [1] Self attention can connect distant tokens directly without passing information step by step through every intermediate position. [2] Self-attention is a mechanism that lets each token compare itself with other tokens in the same sequence and compute a weighted combination of their representations. [3]

### Sources

- [1] faq_attention_003 | attention | knowledge_base/faqs.csv | 0
- [2] faq_attention_006 | attention | knowledge_base/faqs.csv | 0
- [3] self_attention_and_transformer_architecture | attention | knowledge_base/docs/self_attention_and_transformer_architecture.md | 0

### Retrieval Trace Summary

- [1] faq_attention_003: selected as a retrieved candidate
- [2] faq_attention_006: selected as a retrieved candidate
- [3] self_attention_and_transformer_architecture: selected as a retrieved candidate

## 2. Why can transformers train more in parallel than RNNs?

- Rationale: Highlights a classic architectural comparison and retrieval of a conceptual explanation.
- Expected behavior category: `answerable`
- Resolved backend: `tfidf`
- Resolved llm: `offline`
- Latency (ms): 0.90
- Offline fallback used: false

### Answer

Transformers are more parallelizable than RNNs because they process tokens in parallel within a layer instead of stepping strictly one token at a time. [2] RNNs usually process one step after another because each hidden state depends on the previous one. [1] Transformers were designed to handle sequence transduction without relying on recurrence or convolution. [2]

### Sources

- [1] faq_rnns_006 | rnns | knowledge_base/faqs.csv | 0
- [2] transformers_overview | transformers | knowledge_base/docs/transformers_overview.md | 0
- [3] faq_transformers_002 | transformers | knowledge_base/faqs.csv | 0

### Retrieval Trace Summary

- [1] faq_rnns_006: selected as a retrieved candidate
- [2] transformers_overview: selected as a retrieved candidate
- [3] faq_transformers_002: selected as a retrieved candidate

## 3. Why is retrieval helpful before answer generation in a RAG system?

- Rationale: Demonstrates the project's retrieval-augmented workflow on a portfolio-relevant question.
- Expected behavior category: `answerable`
- Resolved backend: `tfidf`
- Resolved llm: `offline`
- Latency (ms): 0.84
- Offline fallback used: false

### Answer

Retrieval augmented generation combines document retrieval with answer generation so the model can condition on external context. [1]

### Sources

- [1] faq_rag_001 | rag | knowledge_base/faqs.csv | 0
- [2] faq_hallucination_faithfulness_001 | hallucination_faithfulness | knowledge_base/faqs.csv | 0
- [3] faq_evaluation_metrics_001 | evaluation_metrics | knowledge_base/faqs.csv | 0

### Retrieval Trace Summary

- [1] faq_rag_001: selected as a retrieved candidate
- [2] faq_hallucination_faithfulness_001: selected as a retrieved candidate
- [3] faq_evaluation_metrics_001: selected as a retrieved candidate

## 4. Why does metadata matter after a vector store retrieves a chunk?

- Rationale: Shows source tracing and citation-oriented evidence handling.
- Expected behavior category: `answerable`
- Resolved backend: `tfidf`
- Resolved llm: `offline`
- Latency (ms): 0.95
- Offline fallback used: false

### Answer

A vector store keeps embeddings and metadata so similarity search can return passages related to a query vector. [1] Metadata lets the system trace a retrieved chunk back to its source topic and other attributes needed for inspection or citation. [2] collection.add stores chunk identifiers documents embeddings and metadata in the collection so they can be queried later. [3]

### Sources

- [1] faq_chromadb_vector_search_001 | chromadb_vector_search | knowledge_base/faqs.csv | 0
- [2] faq_chromadb_vector_search_003 | chromadb_vector_search | knowledge_base/faqs.csv | 0
- [3] faq_chromadb_vector_search_004 | chromadb_vector_search | knowledge_base/faqs.csv | 0

### Retrieval Trace Summary

- [1] faq_chromadb_vector_search_001: selected as a retrieved candidate
- [2] faq_chromadb_vector_search_003: selected as a retrieved candidate
- [3] faq_chromadb_vector_search_004: selected as a retrieved candidate

## 5. What is the capital city of France?

- Rationale: Shows the offline system abstaining on an out-of-scope question.
- Expected behavior category: `out_of_scope`
- Resolved backend: `tfidf`
- Resolved llm: `offline`
- Latency (ms): 0.99
- Offline fallback used: false

### Answer

I do not know based on the retrieved context

### Sources

- [1] neural_networks_basics | neural_networks | knowledge_base/docs/neural_networks_basics.md | 0
- [2] neural_networks_basics | neural_networks | knowledge_base/docs/neural_networks_basics.md | 1
- [3] self_attention_and_transformer_architecture | attention | knowledge_base/docs/self_attention_and_transformer_architecture.md | 0

### Retrieval Trace Summary

- [1] neural_networks_basics: selected as a retrieved candidate
- [2] neural_networks_basics: selected as a retrieved candidate
- [3] self_attention_and_transformer_architecture: selected as a retrieved candidate
