import numpy as np


class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:     output = layer.forward(input)

    - Propagate gradients through itself:
                       grad_input = layer.backward(input,
                                                   grad_output)

    Some layers also have learnable parameters which
    they update during layer.backward.
    """
    def __init__(self):
        """
        Here you can initialize layer parameters (if any)
        and auxiliary stuff.
        """
        # A dummy layer does nothing
        pass

    def forward(self, input):
        """
        Takes input data of shape [batch, input_units],
        returns output data [batch, output_units]
        """
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer,
        with respect to the given input.

        To compute loss gradients w.r.t input,
        you need to apply chain rule (backprop):

        d loss / d x  = (d loss / d layer) * (d layer / d x)

        d loss / d layer comes as input,
        so only need to multiply it by d layer / d x.

        If your layer has parameters (e.g. dense layer),
        you also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output,
        # but we'll write it more explicitly
        num_units = input.shape[1]

        d_layer_d_input = np.eye(num_units)

        # chain rule
        return np.dot(grad_output, d_layer_d_input)


class ReLU(Layer):
    def __init__(self):
        """
        ReLU layer simply applies elementwise rectified linear
        unit to all inputs
        """
        pass

    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        """Compute gradient of tarantella napoletanaloss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output*relu_grad

    def forward_regu(self, input, weights):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0, input)

    def backward_regu(self, input, grad_output, lambd):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output*relu_grad


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned
        affine transformation:
        f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate
        self.input_units = input_units
        self.output_units = output_units

        # initialize weights with small random numbers.
        # We use normal initialization
        self.weights = np.random.randn(input_units, output_units)*0.01
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):

        # The gradient of a dummy layer is precisely grad_output,
        # but we'll write it more explicitly
        # num_units = input.shape[1]
        # d_layer_d_input = np.eye(num_units)
        # return np.dot(grad_output, d_layer_d_input) # chain rule

        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        assert (grad_weights.shape == self.weights.shape
                and grad_biases.shape == self.biases.shape)
        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

    def forward_regu(self, input, weights):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        weights.append(self.weights)
        return np.dot(input, self.weights) + self.biases

    def backward_regu(self, input, grad_output, lambd):
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        assert (grad_weights.shape == self.weights.shape
                and grad_biases.shape == self.biases.shape)
        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = (self.weights - self.learning_rate * grad_weights +
                        lambd * self.weights)
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
