# Self-Attention and Transformer Architecture

Self-attention is a mechanism that lets each token compare itself with other tokens in the same sequence and compute a weighted combination of their representations. This helps the model decide which tokens are most relevant when building context.

Queries, keys, and values are learned vector projections used inside self-attention. A query represents what a token is looking for, keys represent what each token offers, and values hold the information that will be blended into the final contextual representation.

Scaled dot-product attention computes attention scores by taking dot products between queries and keys, scaling them by the square root of the key dimension, applying softmax, and then using the resulting weights to combine the value vectors. The scaling term keeps large dot products from making the softmax distribution too sharp.

Self-attention helps capture long-range relationships because every token can directly attend to every other token in the sequence. Unlike a simple recurrent path, it does not need to pass information through many sequential steps before connecting distant words.

For each token, self-attention produces a contextual output vector. That output mixes information from the token itself and from other relevant tokens, allowing the representation to reflect the broader sentence context.

