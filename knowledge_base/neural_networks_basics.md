# Neural Networks Basics

A perceptron is a simple computational unit that takes weighted inputs, adds a bias term, and produces an output. It is the basic building block that inspired modern neural networks.

Neural networks use activation functions because stacked linear operations would otherwise collapse into a single linear transformation. Activation functions introduce non-linearity, which lets the model learn more complex decision boundaries.

A loss function measures how far the model's prediction is from the expected target. Training tries to minimize this loss so that predictions become more accurate over time.

Backpropagation is the process of computing gradients of the loss with respect to model parameters and then using those gradients to update the weights. This is how a neural network learns from data.

During backpropagation, error information flows backward from the loss through the network so each weight can be adjusted in the direction that reduces future error.

Hidden layers are useful because they allow the network to build intermediate representations. Instead of mapping raw inputs directly to outputs, the model can learn increasingly abstract features at each layer.
