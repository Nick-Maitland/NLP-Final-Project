# Failure Case Report

## Summary

- Evaluated questions: 30
- Answerable questions: 24
- Unanswerable questions: 6
- Recall@3 on answerable questions: 0.92
- MRR@3 on answerable questions: 0.78
- Faithfulness average: 0.87
- Citation valid rate: 1.00
- False abstention rate (answerable): 0.04

## Concrete Weak Examples

### Q21: How do retrieval and metadata together support trustworthy citations in a RAG workflow?
- Failure type: `retrieval`
- Expected source/topic: `faq_rag_002;faq_chromadb_vector_search_003` / `rag;chromadb_vector_search`
- Retrieved sources: `faq_rag_006;faq_transformers_002;faq_chromadb_vector_search_003`
- Retrieved chunks: [1] faq_rag_006: Question: Why is offline fallback useful in a course RAG project? Answer: An offline fallback allows retrieval and ex... || [2] faq_transformers_002: Question: Why are transformers more parallelizable than RNNs? Answer: Tokens within a layer can be processed together... || [3] faq_chromadb_vector_search_003: Question: Why is metadata important in vector search? Answer: Metadata lets the system trace a retrieved chunk back t...
- Answer: I do not know based on the retrieved context.
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context; the abstention decision did not match answerability
- Proposed fix: Tune lexical weighting, add paraphrase coverage, or improve multi-hop candidate fusion.

### Q11: Why is retrieval helpful before answer generation in a RAG system?
- Failure type: `retrieval`
- Expected source/topic: `faq_rag_002` / `rag`
- Retrieved sources: `faq_rag_001;faq_hallucination_faithfulness_001;faq_evaluation_metrics_001`
- Retrieved chunks: [1] faq_rag_001: Question: What is retrieval augmented generation? Answer: Retrieval augmented generation combines document retrieval... || [2] faq_hallucination_faithfulness_001: Question: What is a hallucination in an NLP answer generation system? Answer: A hallucination is content that sounds... || [3] faq_evaluation_metrics_001: Question: What does Recall@3 measure in a retrieval system? Answer: Recall@3 measures whether a relevant source appea...
- Answer: Retrieval augmented generation combines document retrieval with answer generation so the model can condition on external context. [1]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Tune lexical weighting, add paraphrase coverage, or improve multi-hop candidate fusion.

### Q19: How do tensors and automatic differentiation work together during neural network training in PyTorch?
- Failure type: `knowledge_base_coverage`
- Expected source/topic: `tensors_and_pytorch_basics;neural_networks_basics` / `tensors_and_pytorch;neural_networks`
- Retrieved sources: `tensors_and_pytorch_basics;faq_neural_networks_001`
- Retrieved chunks: [1] tensors_and_pytorch_basics: Shape matters because it tells the model how data is organized for computation. PyTorch uses tensors to store model i... || [2] faq_neural_networks_001: Question: What does a neural network learn during training? Answer: A neural network learns parameter values that red... || [3] tensors_and_pytorch_basics: Tensors and PyTorch Basics Tensors are the core data structure used in modern deep learning. A tensor is a generaliza...
- Answer: Automatic differentiation tracks operations on tensors and computes gradients automatically during backpropagation. [1] This makes tensors the basic object that flows through the entire neural network pipeline. [1]
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

### Q03: Why do neural networks need nonlinear activations instead of only linear layers?
- Failure type: `generation`
- Expected source/topic: `neural_networks_basics` / `neural_networks`
- Retrieved sources: `faq_neural_networks_002;neural_networks_basics;faq_neural_networks_006`
- Retrieved chunks: [1] faq_neural_networks_002: Question: Why are non linear activation functions important? Answer: Non linear activations let stacked layers repres... || [2] neural_networks_basics: Neural Networks Basics A perceptron is a simple computational unit that takes weighted inputs, adds a bias term, and... || [3] faq_neural_networks_006: Question: Why can deep networks learn hierarchical features? Answer: Different layers can specialize in different lev...
- Answer: Neural networks use activation functions because stacked linear operations would otherwise collapse into a single linear transformation. [2] Non linear activations let stacked layers represent complex functions that a purely linear model cannot express. [1]
- Why it failed: the answer was only partially grounded in retrieved context
- Proposed fix: Tighten extractive sentence selection and citation filtering so unsupported text is excluded.
