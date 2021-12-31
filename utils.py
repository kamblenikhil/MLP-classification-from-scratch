# Auhtor - Nikhil Kamble
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

from math import log2
import numpy as np

def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative == False:
        return x
    return np.ones(shape=x.shape, dtype=np.float64)

# sigmoid function was referred from this website (cited below) 
# https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    act_func = 1.0 / (1.0 + np.exp(-x))

    if derivative == False:
        return act_func
    return np.exp(-x)/((1+np.exp(-x))**2)

# tanh function was referred from these websites (cited below) 
# https://www.sharpsightlabs.com/blog/numpy-exponential/
# https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    act_func = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    if derivative == False:
        return act_func
    return 1-(act_func)**2

# relu function wa referred from - https://vidyasheela.com/post/relu-activation-function-with-python-code
def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    if derivative == False:
        return 1.0 * (x > 0)
    return x * (x > 0)

# already given
def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))

# cross entropy was referred from - https://machinelearningmastery.com/cross-entropy-for-machine-learning/
def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    return -np.sum(y * np.log(p + 1e-9)) / len(y)

def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    one_hot = np.zeros(shape=(len(y), len(set(y))), dtype=np.int64)
    yT = np.array([y]).T

    for idx, row in enumerate(yT):
        one_hot[idx, row[0]] = 1
    return one_hot