import numpy as np
import matplotlib.pyplot as plt

# Let m = #training examples, n = #number of features
# It takes as input the following: y is R^(m x 1), X is R^(m x n), w is R^(n x 1)
def linear_regression_normal_equation(X, y):
    x1 = np.ones((X.shape[0], 1))
    X = np.append(X, x1, axis=1)
    
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))

    return W

# For visual confirmation, not necessary.
# def plot_fit(W, X, y):
#     random_points = np.linspace(0, 1, 200)[:,np.newaxis]
#     x1 = np.ones((random_points.shape[0], 1))
#     random_points2 = np.append(random_points, x1, axis=1)

#     y_hat = np.dot(random_points2, W)
#     print(y_hat.shape)

#     fig = plt.figure(figsize=(8,6))
#     plt.title("Training set in blue, our hypothesis on the test set in orange")
#     print(X.shape)
#     print(y.shape)
#     plt.scatter(X[:,0], y)

#     plt.scatter(random_points, y_hat)
#     plt.xlabel("First feature")
#     plt.ylabel("Second feature")
#     plt.show()

if __name__ == '__main__':
    m, n = 500, 1
    X = np.random.rand(m, n)
    y = 5*X + np.random.randn(m, n) * 0.1
    #x1 = np.ones((X.shape[0], 1))
    #X = np.append(X, x1, axis=1)
    W = linear_regression_normal_equation(X, y)
