from random import random
from activationFunction import *
from model import *
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Prepare dataset
X, Y = fetch_openml('mnist_784', version=1, data_home="./data/", return_X_y=True)
X = np.array(X/255.0, dtype=np.float32) # Normalize
Y = np.array(Y, dtype=np.uint8)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
train_y = np.eye(10)[train_y].astype(np.int32) # One-hot vector
test_y = np.eye(10)[test_y].astype(np.int32) # One-hot vector
train_n = train_x.shape[0]
test_n = test_x.shape[0]

model = MLP([Linear(784, 500, Swish),
             Linear(500, 500, Swish),
             Linear(500, 10, Softmax)])


def main():
    n_epoch = 20
    batchsize = 100
    lr = 0.5

    for epoch in range(n_epoch):
        print('epoch %d | ' % epoch, end="")

        # Train
        sum_loss = 0
        pred_y = []
        perm = np.random.permutation(train_n)

        for i in range(0, train_n, batchsize):
            x = train_x[perm[i:i+batchsize]]
            t = train_y[perm[i:i+batchsize]]
            sum_loss += model.train(x, t, lr) * len(x)

            pred_y.extend(np.argmax(model.y, axis=1))

        loss = sum_loss / train_n

        # Accuracy
        acc = np.sum(np.eye(10)[pred_y] * train_y[perm]) / train_n
        print('Train loss %.3f, accuracy %.4f | ' %(loss, acc), end="")

        # Test
        sum_loss = 0
        pred_y = []

        for i in range(0, test_n, batchsize):
            x = test_x[i:i+batchsize]
            y = test_y[i:i+batchsize]

            sum_loss += model.test(x, y) * len(x)
            pred_y.extend(np.argmax(model.y, axis=1))

        loss = sum_loss / test_n
        acc = np.sum(np.eye(10)[pred_y] * test_y) / test_n
        print('Test loss %.3f, accuracy %.4f' %(loss, acc))


if __name__ == '__main__':
    main()

