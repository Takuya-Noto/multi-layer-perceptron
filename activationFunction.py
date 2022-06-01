import numpy as np


class Sigmoid:
    def __init__(self):
        self.y = None

    def __call__(self, x): # Forward
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self):
        return self.y * (1 - self.y)

class ReLU:
    def __init__(self):
        self.x = None
    
    def __call__(self, x): # Forward
        self.x = x
        return self.x * (self.x > 0)

    def backward(self):
        return self.x > 0

class Swish:
    def __init__(self):
        self.beta = 1.702 # Approximate GELU
        self.x = None
        self.y = None

    def __call__(self, x): # Forward
        self.x = x
        y = x / (1 + np.exp(-1 * self.beta * x))
        self.y = y
        return y

    def backward(self):
        sigmoid_beta = 1 / (1 + np.exp(-1 * self.beta * self.x))
        return self.beta * self.y + sigmoid_beta * (1 - self.beta * self.y)

class Softmax:
    def __init__(self):
        self.y = None

    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Prevent overflow
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y = y
        return y