"""
Simple two-layered Neural Network from scratch implementation.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-28 Initial coding

"""

import numpy as np
from utils import create_dataset, plot_contour


class NeuralNetwork:
    def __init__(self, X, y):
        # m for #training examples and n for #features
        self.m, self.n = X.shape

        # regularization term lambd (lambda is reserved keyword)
        self.lambd = 1e-3
        self.learning_rate = 0.1

        # Define size of first hidden-layer and second hidden layer (output layer)
        self.h1 = 25
        self.h2 = len(np.unique(y))

    def init_kaiming_weights(self, l0, l1):
        # Kaiming weights
        w = np.random.randn(l0, l1) * np.sqrt(2.0 / l0)
        b = np.zeros((1, l1))

        return w, b

    def forward_prop(self, X, parameters):
        W2 = parameters["W2"]
        W1 = parameters["W1"]
        b2 = parameters["b2"]
        b1 = parameters["b1"]

        # forward prop
        a0 = X
        z1 = np.dot(a0, W1) + b1

        # apply nonlinearity (relu)
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2) + b2

        # softmax on the last layer
        scores = z2
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # cache values from forward pass to use for backward pass
        cache = {"a0": X, "probs": probs, "a1": a1}

        return cache, probs

    def compute_cost(self, y, probs, parameters):
        W2 = parameters["W2"]
        W1 = parameters["W1"]

        y = y.astype(int)
        data_loss = np.sum(-np.log(probs[np.arange(self.m), y]) / self.m)
        reg_loss = 0.5 * self.lambd * np.sum(W1 * W1) + 0.5 * self.lambd * np.sum(
            W2 * W2
        )

        # total cost J
        total_cost = data_loss + reg_loss

        return total_cost

    def back_prop(self, cache, parameters, y):
        # Unpack from parameters
        W2 = parameters["W2"]
        W1 = parameters["W1"]
        b2 = parameters["b2"]
        b1 = parameters["b1"]

        # Unpack from forward prop
        a0 = cache["a0"]
        a1 = cache["a1"]
        probs = cache["probs"]

        dz2 = probs
        dz2[np.arange(self.m), y] -= 1
        dz2 /= self.m

        # backprop through values dW2 and db2
        dW2 = np.dot(a1.T, dz2) + self.lambd * W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Back to the (only) hidden layer in this case
        dz1 = np.dot(dz2, W2.T)
        dz1 = dz1 * (a1 > 0)

        # backprop through values dW1, db1
        dW1 = np.dot(a0.T, dz1) + self.lambd * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {"dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}

        return grads

    def update_parameters(self, parameters, grads):
        learning_rate = self.learning_rate

        W2 = parameters["W2"]
        W1 = parameters["W1"]
        b2 = parameters["b2"]
        b1 = parameters["b1"]

        dW2 = grads["dW2"]
        dW1 = grads["dW1"]
        db2 = grads["db2"]
        db1 = grads["db1"]

        # Do gradient descent step
        W2 -= learning_rate * dW2
        W1 -= learning_rate * dW1
        b2 -= learning_rate * db2
        b1 -= learning_rate * db1

        # store back weights in parameters
        parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

        return parameters

    def main(self, X, y, num_iter=10000):
        # initialize our weights
        W1, b1 = self.init_kaiming_weights(self.n, self.h1)
        W2, b2 = self.init_kaiming_weights(self.h1, self.h2)

        # pack parameters into a dictionary
        parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

        # How many gradient descent updates we want to do
        for it in range(num_iter + 1):

            # forward prop
            cache, probs = self.forward_prop(X, parameters)

            # calculate cost
            cost = self.compute_cost(y, probs, parameters)

            # print cost sometimes
            if it % 2500 == 0:
                print(f"At iteration {it} we have a cost of {cost}")

            # back prop
            grads = self.back_prop(cache, parameters, y)

            # update parameters
            parameters = self.update_parameters(parameters, grads)

        return parameters


if __name__ == "__main__":
    # Generate dataset
    X, y = create_dataset(300, K=3)
    y = y.astype(int)

    # Train network
    NN = NeuralNetwork(X, y)
    trained_parameters = NN.main(X, y)

    # Get trained parameters
    W2 = trained_parameters["W2"]
    W1 = trained_parameters["W1"]
    b2 = trained_parameters["b2"]
    b1 = trained_parameters["b1"]

    # Plot the decision boundary (for nice visualization)
    plot_contour(X, y, NN, trained_parameters)
