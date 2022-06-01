import numpy as np

# Each layer
class Linear:
    def __init__(self, in_dim, out_dim, activation):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation()
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None

    def __call__(self, x):
        # Forward
        self.x = x
        u = np.dot(x, self.W) + self.b
        self.z = self.activation(u)
        return self.z

    def backward(self, dout):
        # Error
        self.delta = dout * self.activation.backward()
        dout = np.dot(self.delta, self.W.T)

        # Gradient
        self.dW = np.dot(self.x.T, self.delta)
        self.db = np.dot(np.ones(len(self.x)), self.delta)

        return dout

# Multi layer perceptron
class MLP():
    def __init__(self, layers):
        self.layers = layers

    def train(self, x, t, lr):
        # Feed forward
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)

        # Loss
        self.loss = np.sum(-t * np.log(self.y + 1e-7)) / len(x)

        # Back propagation
          # Last layer
            # Error, Gradient
        batchsize = len(self.layers[-1].x)
        delta = (self.y - t) / batchsize
        self.layers[-1].delta = delta
        self.layers[-1].dW = np.dot(self.layers[-1].x.T, self.layers[-1].delta)
        self.layers[-1].db = np.dot(np.ones(batchsize), self.layers[-1].delta)
        dout = np.dot(self.layers[-1].delta, self.layers[-1].W.T)

            # Update
        self.layers[-1].W -= lr * self.layers[-1].dW
        self.layers[-1].b -= lr * self.layers[-1].db

          # Middle layers
        for layer in self.layers[-2::-1]:
            # Error, Gradient
            dout = layer.backward(dout)

            # Update
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

        return self.loss

    def test(self, x, t):
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        self.loss = np.sum(-t * np.log(self.y + 1e-7)) / len(x)
        return self.loss

