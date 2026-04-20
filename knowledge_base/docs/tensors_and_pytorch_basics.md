---
title: Tensors and PyTorch Basics
topic: tensors_and_pytorch
kind: doc
---

# Tensors and PyTorch Basics

Tensors are the core data structure used in modern deep learning. A tensor is a generalization of scalars, vectors, and matrices into a single structure that can represent data with any number of dimensions. A scalar is a zero-dimensional tensor, a vector is a one-dimensional tensor, and a matrix is a two-dimensional tensor.

The shape of a tensor describes how many elements exist along each dimension. For example, a tensor with shape `(3, 4)` has 3 rows and 4 columns. Shape matters because it tells the model how data is organized for computation.

PyTorch uses tensors to store model inputs, model parameters, intermediate activations, gradients, and outputs. This makes tensors the basic object that flows through the entire neural network pipeline.

PyTorch is popular because it supports automatic differentiation. Automatic differentiation tracks operations on tensors and computes gradients automatically during backpropagation. This is useful because it allows developers to train neural networks without manually deriving every gradient update.

PyTorch tensors can be moved between devices such as CPU and GPU. Even when GPU acceleration is not available, the same tensor-based programming model works on CPU, which makes local experimentation easier.
