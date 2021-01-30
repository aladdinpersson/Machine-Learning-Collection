"""
Implementation of Linear Regression using the Normal Equation.

Let m = #training examples, n = #number of features and the
input shapes are y is R^(m x 1), X is R^(m x n), w is R^(n x 1).
Using these shapes, the normal equation implementation is
exactly as the derived formula :) 

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-25 Initial coding
"""

import numpy as np


def linear_regression_normal_equation(X, y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1)
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return W


if __name__ == "__main__":
    # Run a small test example: y = 5x (approximately)
    m, n = 500, 1
    X = np.random.rand(m, n)
    y = 5 * X + np.random.randn(m, n) * 0.1
    W = linear_regression_normal_equation(X, y)
