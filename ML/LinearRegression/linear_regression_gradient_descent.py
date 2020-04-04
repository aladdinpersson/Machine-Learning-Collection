import numpy as np

# Let m = #training examples, n = #number of features
# Sizes differ a little bit from blog notation. It takes as input
# the following: y is R^(1 x m), X is R^(n x m), w is R^(n x 1)

class LinearRegression():
    def __init__(self, print_cost=False):
        self.learning_rate = 0.01
        self.total_iterations = 1000
        self.print_cost = print_cost
    
    def y_hat(self, X, w):
        return np.dot(w.T, X)
    
    def cost(self, yhat, y):
        C = 1/self.m * np.sum(np.power(yhat - y, 2))
        
        return C
    
    def gradient_descent(self, w, X, y, yhat):
        dCdW =  2/self.m * np.dot(X, (yhat - y).T)
        w = w - self.learning_rate * dCdW
        
        return w
    
    def main(self, X, y):
        # Add x1 = 1
        x1 = np.ones((1,X.shape[1]))
        X = np.append(X, x1, axis = 0)
        
        self.m = X.shape[1]
        self.n = X.shape[0]
        
        w = np.zeros((self.n, 1))
        
        for it in range(self.total_iterations+1):
            yhat = self.y_hat(X, w)
            cost = self.cost(yhat, y)
            
            if it % 2000 == 0 and self.print_cost:
                print(f'Cost at iteration {it} is {cost}')
            
            w = self.gradient_descent(w, X, y, yhat)
        
        return w
    
    
# For visual confirmation, not necessary.
# def plot_fit(W, X, y):
    # random_points = np.linspace(0, 1, 200)[:,np.newaxis]
    # x1 = np.ones((random_points.shape[0], 1))
    # random_points2 = np.append(random_points, x1, axis=1)
    
    # plt.scatter(X, y)
    # y_hat = np.dot(random_points2, W)
    # print(y_hat.shape)

    # fig = plt.figure(figsize=(8,6))
    # plt.title("Training set in blue, our hypothesis on the test set in orange")
    # print(X.shape)
    # print(y.shape)
    # plt.scatter(X[:,0], y)

    # plt.scatter(random_points, y_hat)
    # plt.xlabel("First feature")
    # plt.ylabel("Second feature")
    # plt.show()

if __name__ == '__main__':
    X = np.random.rand(1, 500)
    y = 3 * X + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.main(X, y)
