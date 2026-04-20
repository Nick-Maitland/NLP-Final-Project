# Failure Case Report

## Summary

- Evaluated questions: 30
- Answerable questions: 24
- Unanswerable questions: 6
- Recall@3 on answerable questions: 0.88
- MRR@3 on answerable questions: 0.69
- Faithfulness average: 0.91
- Citation valid rate: 1.00
- False abstention rate (answerable): 0.00

## Concrete Weak Examples

### Q11: Why is retrieval helpful before answer generation in a RAG system?
- Failure type: `retrieval`
- Expected source/topic: `faq_rag_002` / `rag`
- Retrieved sources: `faq_rag_007;faq_rag_001;faq_rag_009`
- Retrieved chunks: [1] faq_rag_007: Question: Why does retrieval help before answer generation in a grounded QA system? Answer: Retrieval provides specif... || [2] faq_rag_001: Question: What is retrieval augmented generation? Answer: Retrieval augmented generation combines document retrieval... || [3] faq_rag_009: Question: What should a RAG system do when retrieved evidence is weak or off topic? Answer: A grounded system should...
- Answer: Retrieval provides specific passages before generation so the answer can rely on evidence instead of only on model memory. [1]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Tune lexical weighting, add paraphrase coverage, or improve multi-hop candidate fusion.

### Q21: How do retrieval and metadata together support trustworthy citations in a RAG workflow?
- Failure type: `retrieval`
- Expected source/topic: `faq_rag_002;faq_chromadb_vector_search_003` / `rag;chromadb_vector_search`
- Retrieved sources: `faq_rag_008;faq_chromadb_vector_search_008;faq_hallucination_faithfulness_007`
- Retrieved chunks: [1] faq_rag_008: Question: How do retrieved passages support trustworthy citations in a RAG workflow? Answer: Retrieved passages give... || [2] faq_chromadb_vector_search_008: Question: Why do vector databases store both embeddings and metadata? Answer: Embeddings support similarity search wh... || [3] faq_hallucination_faithfulness_007: Question: What is retrieval confidence in a grounded QA system? Answer: Retrieval confidence is an estimate of whethe...
- Answer: Embeddings support similarity search while metadata keeps the results traceable and useful for filtering inspection and citation. [2]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Tune lexical weighting, add paraphrase coverage, or improve multi-hop candidate fusion.

### Q23: Why are Recall@3 and faithfulness both needed when evaluating a grounded QA system?
- Failure type: `knowledge_base_coverage`
- Expected source/topic: `faq_evaluation_metrics_001;faq_evaluation_metrics_003` / `evaluation_metrics;evaluation_metrics`
- Retrieved sources: `faq_evaluation_metrics_006;faq_evaluation_metrics_003;faq_hallucination_faithfulness_007`
- Retrieved chunks: [1] faq_evaluation_metrics_006: Question: Why are Recall@3 and faithfulness both needed in grounded QA evaluation? Answer: Recall@3 checks whether us... || [2] faq_evaluation_metrics_003: Question: What is a faithfulness score in a grounded QA system? Answer: A faithfulness score estimates how well the g... || [3] faq_hallucination_faithfulness_007: Question: What is retrieval confidence in a grounded QA system? Answer: Retrieval confidence is an estimate of whethe...
- Answer: Recall@3 checks whether useful evidence was retrieved while faithfulness checks whether the final answer stayed supported by that evidence. [1]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Expand the knowledge base with clearer passages for this concept or domain.

### Q19: How do tensors and automatic differentiation work together during neural network training in PyTorch?
- Failure type: `knowledge_base_coverage`
- Expected source/topic: `tensors_and_pytorch_basics;neural_networks_basics` / `tensors_and_pytorch;neural_networks`
- Retrieved sources: `faq_tensors_and_pytorch_002;tensors_and_pytorch_basics;faq_tensors_and_pytorch_001`
- Retrieved chunks: [1] faq_tensors_and_pytorch_002: Question: How does automatic differentiation work with tensors in PyTorch? Answer: Autograd tracks tensor operations... || [2] tensors_and_pytorch_basics: Shape matters because it tells the model how data is organized for computation. PyTorch uses tensors to store model i... || [3] faq_tensors_and_pytorch_001: Question: Why are tensors central to neural network training in PyTorch? Answer: Tensors hold inputs parameters activ...
- Answer: Automatic differentiation tracks operations on tensors and computes gradients automatically during backpropagation. [2] This makes tensors the basic object that flows through the entire neural network pipeline. [2]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Expand the knowledge base with clearer passages for this concept or domain.

### Q07: Why can transformers train more in parallel than ordinary RNNs?
- Failure type: `generation`
- Expected source/topic: `faq_transformers_002` / `transformers`
- Retrieved sources: `faq_rnns_006;transformers_overview;faq_transformers_002`
- Retrieved chunks: [1] faq_rnns_006: Question: Why are RNNs less parallelizable than transformers? Answer: RNNs usually process one step after another bec... || [2] transformers_overview: Transformers Overview Transformers were designed to handle sequence transduction without relying on recurrence or con... || [3] faq_transformers_002: Question: Why are transformers more parallelizable than RNNs? Answer: Tokens within a layer can be processed together...
- Answer: Transformers are more parallelizable than RNNs because they process tokens in parallel within a layer instead of stepping strictly one token at a time. [2]
- Why it failed: the answer was only partially grounded in retrieved context
- Proposed fix: Tighten extractive sentence selection and citation filtering so unsupported text is excluded.
