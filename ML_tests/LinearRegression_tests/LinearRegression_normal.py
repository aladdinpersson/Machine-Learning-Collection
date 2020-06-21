# Import folder where sorting algorithms
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("ML/algorithms/linearregression")

# If run from local:
# sys.path.append('../../ML/algorithms/linearregression/')
from linear_regression_normal_equation import linear_regression_normal_equation


class TestLinearRegression_NormalEq(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.X1 = np.array([[0, 1, 2]]).T
        self.y1 = np.array([1, 2, 3])
        self.W1_correct = np.array([[1, 1]])

        self.X2 = np.array([[0, 1]]).T
        self.y2 = np.array([1, 0])
        self.W2_correct = np.array([[1, -1]])

        self.X3 = np.array([[1, 2, 3], [1, 2, 4]]).T
        self.y3 = np.array([5, 10, 18])
        self.W3_correct = np.array([[0, 2, 3]])

        self.X4 = np.array([[0, 0]]).T
        self.y4 = np.array([0, 0])
        self.W4_correct = np.array([[0, 0]])

        self.X5 = np.array([[0, 1, 2, 3, 4, 5]]).T
        self.y5 = np.array([0, 0.99, 2.01, 2.99, 4.01, 4.99])
        self.W5_correct = np.array([[0, 1]])

    def test_perfectpositiveslope(self):
        W = linear_regression_normal_equation(self.X1, self.y1)
        print(W.shape)
        print(self.W1_correct.shape)
        boolean_array = np.isclose(W, self.W1_correct)
        self.assertTrue(boolean_array.all())

    def test_perfectnegativeslope(self):
        W = linear_regression_normal_equation(self.X2, self.y2)
        boolean_array = np.isclose(W, self.W2_correct)
        self.assertTrue(boolean_array.all())

    def test_multipledimension(self):
        W = linear_regression_normal_equation(self.X3, self.y3)
        print(W)
        print(self.W3_correct)
        boolean_array = np.isclose(W, self.W3_correct)
        self.assertTrue(boolean_array.all())

    def test_zeros(self):
        W = linear_regression_normal_equation(self.X4, self.y4)
        boolean_array = np.isclose(W, self.W4_correct)
        self.assertTrue(boolean_array.all())

    def test_noisydata(self):
        W = linear_regression_normal_equation(self.X5, self.y5)
        boolean_array = np.isclose(W, self.W5_correct, atol=1e-3)
        self.assertTrue(boolean_array.all())


if __name__ == "__main__":
    print("Running Linear Regression Normal Equation tests:")
    unittest.main()
