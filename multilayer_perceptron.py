# Auhtor - Nikhil Kamble
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

# Samsun Zang Youtube channel videos were helpful - https://www.youtube.com/channel/UCNyoJQcV6NCHiRCeqcCxJag
# https://youtu.be/CqOfi41LfDw - This video really helped me a lot, to understand Neural Networks and backpropagation
# CSCI B551 QnA community, was also helpful to solve some of my doubts

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = one_hot_encoding(y)

        # streamline the specific train/test data at runtime 
        np.random.seed(42)

        # initializing the parameters for neural network - hidden and output layer (features, weights and biases)

        n_hidden = self.n_hidden
        n_samples = y.shape[0]
        n_output = len(set(y))
        n_features = X.shape[1]
        h_weight = np.random.rand(n_features, n_hidden)
        o_weight = np.random.rand(n_hidden, n_output)

        self._h_weights = h_weight
        shape_h = []

        for i in range(n_hidden):
            shape_h.append(1.0)

        self._h_bias = shape_h

        self._o_weights = o_weight
        shape_o = []

        for i in range(n_output):
            shape_o.append(1.0)

        self._o_bias = shape_o

    def fit(self, X, y):
        self._initialize(X, y)
        learning_rate = self.learning_rate
        i = 0
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        while(i != self.n_iterations):
            #i = self.n_iterations
            
            ##### feed forward
            Z0, A0, Z1, A1 = self.ff(self._X)

            ##### back propagation

            # output layer ---> errors
            op_delta = np.multiply((self._y - A1), (self._output_activation(Z1, derivative=True)))

            # hidden layer ---> errors
            hd_delta = np.multiply((np.dot(op_delta, self._o_weights.T)), (self.hidden_activation(Z0, derivative=True)))

            # biases upgradation
            self._o_bias += np.sum(op_delta, axis=0) * learning_rate
            self._h_bias += np.sum(hd_delta, axis=0) * learning_rate

            # weights upgradation
            self._h_weights += np.dot(self._X.T, hd_delta) * learning_rate
            self._o_weights += np.dot(A0.T, op_delta) * learning_rate

            ##### after every 20 iterations stores the cross-entropy loss

            if i % 20 == 0:
                self._loss_history.append(self._loss_function(self._y, A1))

            i += 1

    ##### prediction on the test dataset

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        Z0, A0, Z1, A1 = self.ff(X)
        return np.array([np.argmax(x) for x in A1])

    ##### feed forward

    def ff(self, z):

        # assigning the bias to hidden and output layer
        hidden_b = self._h_bias
        output_b = self._o_bias

        # Z0 = _X(dot product)hidden_weight + hidden_bias
        Z0 = np.dot(z, self._h_weights) + hidden_b
        
        # A0 = g(Z0) - hidden_weight
        A0 = self.hidden_activation(Z0)

        # Z1 = A0(dot product)output_weight + output_bias
        Z1 = np.dot(A0, self._o_weights) + output_b

        # A1 = g(Z1)
        A1 = self._output_activation(Z1)

        return Z0, A0, Z1, A1