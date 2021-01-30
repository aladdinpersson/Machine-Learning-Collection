# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from googlenet import GoogLeNet

if __name__ == "__main__":
    model = GoogLeNet(input_shape = (224, 224, 3))
    model.summary()