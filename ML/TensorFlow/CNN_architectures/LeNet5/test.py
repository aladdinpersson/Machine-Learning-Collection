# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from lenet5 import LeNet5

if __name__ == "__main__":
    model = LeNet5(input_shape = (32, 32, 1), classes = 10)
    model.summary()