# Evaluation Report

## Summary Metrics

- Questions: 30
- Answerable: 24
- Unanswerable: 6
- Retrieval Recall@3 (answerable): 0.92
- MRR@3 (answerable): 0.78
- Faithfulness average: 0.87
- Citation valid rate: 1.00
- Abstention accuracy (unanswerable): 1.0
- False abstention rate (answerable): 0.04
- Average latency (ms): 1.08
- Median latency (ms): 1.09

## Topic Breakdown

- `attention`: questions=3, answerable=3, recall@3=1.0, faithfulness=0.91
- `bert_style_encoders`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.89
- `chromadb_vector_search`: questions=2, answerable=2, recall@3=0.75, faithfulness=0.67
- `embeddings`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.89
- `evaluation_metrics`: questions=3, answerable=3, recall@3=1.0, faithfulness=0.86
- `gpt_style_decoders`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.9
- `hallucination_faithfulness`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.92
- `lstms_grus`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.82
- `neural_networks`: questions=2, answerable=2, recall@3=0.75, faithfulness=0.81
- `out_of_scope`: questions=6, answerable=0, recall@3=None, faithfulness=0.93
- `rag`: questions=2, answerable=2, recall@3=0.25, faithfulness=0.67
- `rnns`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.87
- `rnns_training`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.94
- `sequence_to_sequence`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.94
- `tensors_and_pytorch`: questions=2, answerable=2, recall@3=0.75, faithfulness=0.96
- `tokenization`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.87
- `transformers`: questions=3, answerable=3, recall@3=1.0, faithfulness=0.86

## Analysis

- Multi-hop questions: 6
- Multi-hop Recall@3 (answerable): 0.83
- Paraphrased questions: 17
- Unanswerable questions answered incorrectly: 0
- Citation-valid answers: 1.00
- False abstentions on answerable questions: 0.04
- Weakest answerable topics by recall: rag (0.25), chromadb_vector_search (0.75), neural_networks (0.75)
- Weak areas are reflected directly in the concrete examples below rather than being smoothed away.
- Imperfect results are expected on paraphrase and multi-hop rows because they stress retrieval coverage and grounded composition.

## Weak Examples

### Q21: How do retrieval and metadata together support trustworthy citations in a RAG workflow?
- Notes: multi-hop;paraphrase
- Recall@3: 0.5
- Reciprocal rank: 0.33
- Faithfulness: 0.60
- Citation valid: true
- Abstention correct: false
- Retrieved chunks: [1] faq_rag_006: Question: Why is offline fallback useful in a course RAG project? Answer: An offline fallback allows retrieval and ex... || [2] faq_transformers_002: Question: Why are transformers more parallelizable than RNNs? Answer: Tokens within a layer can be processed together... || [3] faq_chromadb_vector_search_003: Question: Why is metadata important in vector search? Answer: Metadata lets the system trace a retrieved chunk back t...
- Answer: I do not know based on the retrieved context.

### Q11: Why is retrieval helpful before answer generation in a RAG system?
- Notes: single-hop;paraphrase
- Recall@3: 0.0
- Reciprocal rank: 0.0
- Faithfulness: 0.73
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_rag_001: Question: What is retrieval augmented generation? Answer: Retrieval augmented generation combines document retrieval... || [2] faq_hallucination_faithfulness_001: Question: What is a hallucination in an NLP answer generation system? Answer: A hallucination is content that sounds... || [3] faq_evaluation_metrics_001: Question: What does Recall@3 measure in a retrieval system? Answer: Recall@3 measures whether a relevant source appea...
- Answer: Retrieval augmented generation combines document retrieval with answer generation so the model can condition on external context. [1]

### Q19: How do tensors and automatic differentiation work together during neural network training in PyTorch?
- Notes: multi-hop;doc
- Recall@3: 0.5
- Reciprocal rank: 1.0
- Faithfulness: 0.92
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] tensors_and_pytorch_basics: Shape matters because it tells the model how data is organized for computation. PyTorch uses tensors to store model i... || [2] faq_neural_networks_001: Question: What does a neural network learn during training? Answer: A neural network learns parameter values that red... || [3] tensors_and_pytorch_basics: Tensors and PyTorch Basics Tensors are the core data structure used in modern deep learning. A tensor is a generaliza...
- Answer: Automatic differentiation tracks operations on tensors and computes gradients automatically during backpropagation. [1] This makes tensors the basic object that flows through the entire neural network pipeline. [1] PyTorch is popular because it supports automatic differentiation. [1]

### Q07: Why can transformers train more in parallel than ordinary RNNs?
- Notes: single-hop;paraphrase
- Recall@3: 1.0
- Reciprocal rank: 0.33
- Faithfulness: 0.78
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_rnns_006: Question: Why are RNNs less parallelizable than transformers? Answer: RNNs usually process one step after another bec... || [2] transformers_overview: Transformers Overview Transformers were designed to handle sequence transduction without relying on recurrence or con... || [3] faq_transformers_002: Question: Why are transformers more parallelizable than RNNs? Answer: Tokens within a layer can be processed together...
- Answer: Transformers are more parallelizable than RNNs because they process tokens in parallel within a layer instead of stepping strictly one token at a time. [2] RNNs usually process one step after another because each hidden state depends on the previous one. [1] Transformers were designed to handle sequence transduction without relying on recurrence or convolution. [2]

### Q03: Why do neural networks need nonlinear activations instead of only linear layers?
- Notes: single-hop;paraphrase
- Recall@3: 1.0
- Reciprocal rank: 0.5
- Faithfulness: 0.70
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_neural_networks_002: Question: Why are non linear activation functions important? Answer: Non linear activations let stacked layers repres... || [2] neural_networks_basics: Neural Networks Basics A perceptron is a simple computational unit that takes weighted inputs, adds a bias term, and... || [3] faq_neural_networks_006: Question: Why can deep networks learn hierarchical features? Answer: Different layers can specialize in different lev...
- Answer: Non linear activations let stacked layers represent complex functions that a purely linear model cannot express. [1] Neural networks use activation functions because stacked linear operations would otherwise collapse into a single linear transformation. [2] It is the basic building block that inspired modern neural networks. [2]
