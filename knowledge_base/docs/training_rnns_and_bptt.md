---
title: Training RNNs and Backpropagation Through Time
topic: rnns_training
kind: doc
---

# Training RNNs and Backpropagation Through Time

Backpropagation through time, or BPTT, is the extension of backpropagation to recurrent neural networks. The RNN is conceptually unrolled across time steps, and gradients are propagated backward through each step of the sequence.

Long sequences can cause vanishing gradients because repeated multiplication by small derivatives shrinks the gradient signal as it moves backward through time. When that happens, the model struggles to learn long-range dependencies.

Gradient clipping helps RNN training by limiting the size of very large gradients. This reduces unstable updates and helps prevent exploding gradients during training.

Teacher forcing is a training strategy in which the true previous target token is fed into the decoder during training instead of the model's own earlier prediction. This can stabilize and speed up training.

Truncating very long sequences is sometimes useful because it reduces memory cost and makes optimization more manageable. It is a practical compromise when full unrolling would be too expensive.
