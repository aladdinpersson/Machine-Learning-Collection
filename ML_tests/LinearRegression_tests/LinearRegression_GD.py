# Import folder where sorting algorithms
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("ML/algorithms/linearregression")
# If run from local:
# sys.path.append('../../ML/algorithms/linearregression')
from linear_regression_gradient_descent import LinearRegression


class TestLinearRegression_GradientDescent(unittest.TestCase):
    def setUp(self):
        # test cases we want to run

        self.linearReg = LinearRegression()
        self.X1 = np.array([[0, 1, 2]])
        self.y1 = np.array([[1, 2, 3]])
        self.W1_correct = np.array([[1, 1]]).T

        self.X2 = np.array([[0, 1]])
        self.y2 = np.array([[1, 0]])
        self.W2_correct = np.array([[1, -1]]).T

        self.X3 = np.array([[1, 2, 3], [1, 2, 4]])
        self.y3 = np.array([[5, 10, 18]])
        self.W3_correct = np.array([[0, 2, 3]]).T

        self.X4 = np.array([[0, 0]])
        self.y4 = np.array([[0, 0]])
        self.W4_correct = np.array([[0, 0]]).T

        self.X5 = np.array([[0, 1, 2, 3, 4, 5]])
        self.y5 = np.array([[0, 0.99, 2.01, 2.99, 4.01, 4.99]])
        self.W5_correct = np.array([[0, 1]]).T

    def test_perfectpositiveslope(self):
        W = self.linearReg.main(self.X1, self.y1)
        boolean_array = np.isclose(W, self.W1_correct, atol=0.1)
        self.assertTrue(boolean_array.all())

    def test_perfectnegativeslope(self):
        W = self.linearReg.main(self.X2, self.y2)
        boolean_array = np.isclose(W, self.W2_correct, atol=0.1)
        self.assertTrue(boolean_array.all())

    def test_multipledimension(self):
        W = self.linearReg.main(self.X3, self.y3)
        boolean_array = np.isclose(W, self.W3_correct, atol=0.1)
        self.assertTrue(boolean_array.all())

    def test_zeros(self):
        W = self.linearReg.main(self.X4, self.y4)
        boolean_array = np.isclose(W, self.W4_correct, atol=0.1)
        self.assertTrue(boolean_array.all())

    def test_noisydata(self):
        W = self.linearReg.main(self.X5, self.y5)
        boolean_array = np.isclose(W, self.W5_correct, atol=0.1)
        self.assertTrue(boolean_array.all())


if __name__ == "__main__":
    print("Running Linear Regression Normal Equation tests:")
    unittest.main()
