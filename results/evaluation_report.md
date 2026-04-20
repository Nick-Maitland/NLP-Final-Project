# Evaluation Report

## Summary Metrics

- Questions: 30
- Answerable: 24
- Unanswerable: 6
- Retrieval Recall@3 (answerable): 0.88
- MRR@3 (answerable): 0.69
- Faithfulness average: 0.91
- Citation valid rate: 1.00
- Abstention accuracy (unanswerable): 1.0
- False abstention rate (answerable): 0.0
- Average latency (ms): 1.53
- Median latency (ms): 1.47

## Topic Breakdown

- `attention`: questions=3, answerable=3, recall@3=1.0, faithfulness=0.94
- `bert_style_encoders`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.89
- `chromadb_vector_search`: questions=2, answerable=2, recall@3=0.5, faithfulness=0.94
- `embeddings`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.89
- `evaluation_metrics`: questions=3, answerable=3, recall@3=0.67, faithfulness=0.92
- `gpt_style_decoders`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.9
- `hallucination_faithfulness`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.92
- `lstms_grus`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.82
- `neural_networks`: questions=2, answerable=2, recall@3=0.75, faithfulness=0.92
- `out_of_scope`: questions=6, answerable=0, recall@3=None, faithfulness=0.93
- `rag`: questions=2, answerable=2, recall@3=0.0, faithfulness=0.94
- `rnns`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.87
- `rnns_training`: questions=2, answerable=2, recall@3=1.0, faithfulness=0.94
- `sequence_to_sequence`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.94
- `tensors_and_pytorch`: questions=2, answerable=2, recall@3=0.75, faithfulness=0.92
- `tokenization`: questions=1, answerable=1, recall@3=1.0, faithfulness=0.87
- `transformers`: questions=3, answerable=3, recall@3=1.0, faithfulness=0.89

## Analysis

- Multi-hop questions: 6
- Multi-hop Recall@3 (answerable): 0.67
- Paraphrased questions: 17
- Unanswerable questions answered incorrectly: 0
- Citation-valid answers: 1.00
- False abstentions on answerable questions: 0.0
- Weakest answerable topics by recall: rag (0.00), chromadb_vector_search (0.50), evaluation_metrics (0.67)
- Weak areas are reflected directly in the concrete examples below rather than being smoothed away.
- Imperfect results are expected on paraphrase and multi-hop rows because they stress retrieval coverage and grounded composition.

## Weak Examples

### Q11: Why is retrieval helpful before answer generation in a RAG system?
- Notes: single-hop;paraphrase
- Recall@3: 0.0
- Reciprocal rank: 0.0
- Faithfulness: 0.94
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_rag_007: Question: Why does retrieval help before answer generation in a grounded QA system? Answer: Retrieval provides specif... || [2] faq_rag_001: Question: What is retrieval augmented generation? Answer: Retrieval augmented generation combines document retrieval... || [3] faq_rag_009: Question: What should a RAG system do when retrieved evidence is weak or off topic? Answer: A grounded system should...
- Answer: Retrieval provides specific passages before generation so the answer can rely on evidence instead of only on model memory. [1]

### Q21: How do retrieval and metadata together support trustworthy citations in a RAG workflow?
- Notes: multi-hop;paraphrase
- Recall@3: 0.0
- Reciprocal rank: 0.0
- Faithfulness: 0.95
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_rag_008: Question: How do retrieved passages support trustworthy citations in a RAG workflow? Answer: Retrieved passages give... || [2] faq_chromadb_vector_search_008: Question: Why do vector databases store both embeddings and metadata? Answer: Embeddings support similarity search wh... || [3] faq_hallucination_faithfulness_007: Question: What is retrieval confidence in a grounded QA system? Answer: Retrieval confidence is an estimate of whethe...
- Answer: Embeddings support similarity search while metadata keeps the results traceable and useful for filtering inspection and citation. [2]

### Q23: Why are Recall@3 and faithfulness both needed when evaluating a grounded QA system?
- Notes: multi-hop;paraphrase
- Recall@3: 0.5
- Reciprocal rank: 0.5
- Faithfulness: 0.95
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_evaluation_metrics_006: Question: Why are Recall@3 and faithfulness both needed in grounded QA evaluation? Answer: Recall@3 checks whether us... || [2] faq_evaluation_metrics_003: Question: What is a faithfulness score in a grounded QA system? Answer: A faithfulness score estimates how well the g... || [3] faq_hallucination_faithfulness_007: Question: What is retrieval confidence in a grounded QA system? Answer: Retrieval confidence is an estimate of whethe...
- Answer: Recall@3 checks whether useful evidence was retrieved while faithfulness checks whether the final answer stayed supported by that evidence. [1]

### Q19: How do tensors and automatic differentiation work together during neural network training in PyTorch?
- Notes: multi-hop;doc
- Recall@3: 0.5
- Reciprocal rank: 0.5
- Faithfulness: 0.96
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_tensors_and_pytorch_002: Question: How does automatic differentiation work with tensors in PyTorch? Answer: Autograd tracks tensor operations... || [2] tensors_and_pytorch_basics: Shape matters because it tells the model how data is organized for computation. PyTorch uses tensors to store model i... || [3] faq_tensors_and_pytorch_001: Question: Why are tensors central to neural network training in PyTorch? Answer: Tensors hold inputs parameters activ...
- Answer: Automatic differentiation tracks operations on tensors and computes gradients automatically during backpropagation. [2] This makes tensors the basic object that flows through the entire neural network pipeline. [2]

### Q07: Why can transformers train more in parallel than ordinary RNNs?
- Notes: single-hop;paraphrase
- Recall@3: 1.0
- Reciprocal rank: 0.33
- Faithfulness: 0.78
- Citation valid: true
- Abstention correct: true
- Retrieved chunks: [1] faq_rnns_006: Question: Why are RNNs less parallelizable than transformers? Answer: RNNs usually process one step after another bec... || [2] transformers_overview: Transformers Overview Transformers were designed to handle sequence transduction without relying on recurrence or con... || [3] faq_transformers_002: Question: Why are transformers more parallelizable than RNNs? Answer: Tokens within a layer can be processed together...
- Answer: Transformers are more parallelizable than RNNs because they process tokens in parallel within a layer instead of stepping strictly one token at a time. [2]
