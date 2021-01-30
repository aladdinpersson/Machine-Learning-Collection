# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from resnet import ResNet

if __name__ == "__main__":
    # test ResNet50
    model = ResNet(name = "Resnet50", layers = [3, 4, 6, 3], input_shape = (64, 64, 3), classes = 6)
    model.summary()
