---
title: Transformers Overview
topic: transformers
kind: doc
---

# Transformers Overview

Transformers were designed to handle sequence transduction without relying on recurrence or convolution. They use attention mechanisms to model relationships between tokens and to make training more parallelizable than RNN-based systems.

Transformers are more parallelizable than RNNs because they process tokens in parallel within a layer instead of stepping strictly one token at a time. This improves training efficiency on modern hardware.

Positional encodings add information about token order to the model. Because self-attention alone does not inherently know which token came first, positional information tells the transformer where each token appears in the sequence.

Multi-head attention allows the model to attend to different relationships at the same time. Each head can learn a different pattern, such as local syntax, long-range dependency, or role-specific interactions.

In an encoder-decoder transformer, the encoder builds contextual representations of the input sequence and the decoder uses those representations to generate the output sequence. This architecture is common in sequence-to-sequence tasks such as translation.
