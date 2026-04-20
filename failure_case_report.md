# Failure Case Report

## Summary

- Evaluated questions: 30
- Answerable questions: 24
- Unanswerable questions: 6
- Recall@3 on answerable questions: 0.92
- MRR@3 on answerable questions: 0.78
- Faithfulness average: 0.85
- Citation valid rate: 1.00

## Concrete Weak Examples

### Q11: Why is retrieval helpful before answer generation in a RAG system?
- Failure type: `retrieval`
- Expected source/topic: `faq_rag_002` / `rag`
- Retrieved sources: `faq_rag_001;faq_hallucination_faithfulness_001;faq_evaluation_metrics_001`
- Retrieved chunks: [1] faq_rag_001: Question: What is retrieval augmented generation? Answer: Retrieval augmented generation combines document retrieval... || [2] faq_hallucination_faithfulness_001: Question: What is a hallucination in an NLP answer generation system? Answer: A hallucination is content that sounds... || [3] faq_evaluation_metrics_001: Question: What does Recall@3 measure in a retrieval system? Answer: Recall@3 measures whether a relevant source appea...
- Answer: Retrieval augmented generation combines document retrieval with answer generation so the model can condition on external context. [1]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Tune lexical weighting, add paraphrase coverage, or improve multi-hop candidate fusion.

### Q26: Who founded PyTorch?
- Failure type: `generation`
- Expected source/topic: `none` / `out_of_scope`
- Retrieved sources: `tensors_and_pytorch_basics;neural_networks_basics`
- Retrieved chunks: [1] tensors_and_pytorch_basics: Shape matters because it tells the model how data is organized for computation. PyTorch uses tensors to store model i... || [2] tensors_and_pytorch_basics: Tensors and PyTorch Basics Tensors are the core data structure used in modern deep learning. A tensor is a generaliza... || [3] neural_networks_basics: Neural Networks Basics A perceptron is a simple computational unit that takes weighted inputs, adds a bias term, and...
- Answer: PyTorch is popular because it supports automatic differentiation. [1] PyTorch tensors can be moved between devices such as CPU and GPU. [1] PyTorch uses tensors to store model inputs, model parameters, intermediate activations, gradients, and outputs. [1]
- Why it failed: the answer was only partially grounded in retrieved context; the abstention decision did not match answerability
- Proposed fix: Tighten extractive sentence selection and citation filtering so unsupported text is excluded.

### Q21: How do retrieval and metadata together support trustworthy citations in a RAG workflow?
- Failure type: `generation`
- Expected source/topic: `faq_rag_002;faq_chromadb_vector_search_003` / `rag;chromadb_vector_search`
- Retrieved sources: `faq_rag_006;faq_transformers_002;faq_chromadb_vector_search_003`
- Retrieved chunks: [1] faq_rag_006: Question: Why is offline fallback useful in a course RAG project? Answer: An offline fallback allows retrieval and ex... || [2] faq_transformers_002: Question: Why are transformers more parallelizable than RNNs? Answer: Tokens within a layer can be processed together... || [3] faq_chromadb_vector_search_003: Question: Why is metadata important in vector search? Answer: Metadata lets the system trace a retrieved chunk back t...
- Answer: An offline fallback allows retrieval and extractive answering to keep working even when API access or dense model downloads are unavailable. [1] Tokens within a layer can be processed together because attention does not require a strictly step by step hidden state update. [2] Metadata lets the system trace a retrieved chunk back to its source topic and other attributes needed for inspection or citation. [3]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Tighten extractive sentence selection and citation filtering so unsupported text is excluded.

### Q27: In what year was BERT first released?
- Failure type: `generation`
- Expected source/topic: `none` / `out_of_scope`
- Retrieved sources: `faq_bert_style_encoders_001;faq_bert_style_encoders_004;transformers_overview`
- Retrieved chunks: [1] faq_bert_style_encoders_001: Question: What is a BERT style encoder? Answer: A BERT style encoder is a bidirectional transformer encoder that buil... || [2] faq_bert_style_encoders_004: Question: What does bidirectional context mean in BERT like models? Answer: It means a token representation can depen... || [3] transformers_overview: Because self-attention alone does not inherently know which token came first, positional information tells the transf...
- Answer: A BERT style encoder is a bidirectional transformer encoder that builds contextual representations from both left and right context. [1] Because self-attention alone does not inherently know which token came first, positional information tells the transformer where each token appears in the sequence. [3]
- Why it failed: the answer was only partially grounded in retrieved context; the abstention decision did not match answerability
- Proposed fix: Tighten extractive sentence selection and citation filtering so unsupported text is excluded.

### Q19: How do tensors and automatic differentiation work together during neural network training in PyTorch?
- Failure type: `knowledge_base_coverage`
- Expected source/topic: `tensors_and_pytorch_basics;neural_networks_basics` / `tensors_and_pytorch;neural_networks`
- Retrieved sources: `tensors_and_pytorch_basics;faq_neural_networks_001`
- Retrieved chunks: [1] tensors_and_pytorch_basics: Shape matters because it tells the model how data is organized for computation. PyTorch uses tensors to store model i... || [2] faq_neural_networks_001: Question: What does a neural network learn during training? Answer: A neural network learns parameter values that red... || [3] tensors_and_pytorch_basics: Tensors and PyTorch Basics Tensors are the core data structure used in modern deep learning. A tensor is a generaliza...
- Answer: Automatic differentiation tracks operations on tensors and computes gradients automatically during backpropagation. [1] This makes tensors the basic object that flows through the entire neural network pipeline. [1] PyTorch is popular because it supports automatic differentiation. [1]
- Why it failed: expected evidence was not fully retrieved in the top 3; the answer was only partially grounded in retrieved context
- Proposed fix: Expand the knowledge base with clearer passages for this concept or domain.
