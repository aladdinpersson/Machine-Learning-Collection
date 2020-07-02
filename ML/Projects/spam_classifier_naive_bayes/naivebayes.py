"""
Naive Bayes Classifier Implementation from scratch

To run the code structure the code in the following way:
    X be size: (num_training_examples, num_features)
    y be size: (num_classes, )

Where the classes are 0, 1, 2, etc. Then an example run looks like:
    NB = NaiveBayes(X, y)
    NB.fit(X)
    predictions = NB.predict(X)

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-21 Initial coding

"""
import numpy as np


class NaiveBayes:
    def __init__(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self, X):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(
                X, self.classes_mean[str(c)], self.classes_variance[str(c)]
            )
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs


if __name__ == "__main__":
    # For spam emails (Make sure to run build_vocab etc. to have .npy files)
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    NB = NaiveBayes(X, y)
    NB.fit(X)
    y_pred = NB.predict(X)

    print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")
