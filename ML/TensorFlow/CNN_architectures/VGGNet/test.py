# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from vggnet import VGGNet

# Integer value represents output channel after performing the convolution layer
# 'M' represents the max pooling layer
# After convolution blocks; flatten the output and use 4096x4096x1000 Linear Layers
# with soft-max at the end
VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

if __name__ == "__main__":
    # test VGGNet16
    model = VGGNet(name = "VGGNet16", architecture = VGG_types["VGG16"], input_shape=(224, 224, 3), classes = 1000)
    model.summary()
