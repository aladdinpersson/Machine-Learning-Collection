# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.optimizers import Adam
from vggnet import VGGNet

import numpy as np

# Integer value represents output channel after performing the convolution layer
# 'M' represents the max pooling layer
# After convolution blocks; flatten the output and use 4096x4096x1000 Linear Layers
# with soft-max at the end
# NOTE: 1000 is the number of classes for prediction
VGG11 = [
         64, "M",            # Layer 1
         128, "M",           # Layer 2
    256, 256, "M",           # Layer 3
    512, 512, "M",           # Layer 4
    512, 512, "M",           # Layer 5
]
VGG13 = [
     64,  64, 'M',           # Layer 1
    128, 128, 'M',           # Layer 2
    256, 256, "M",           # Layer 3
    512, 512, "M",           # Layer 4
    512, 512, "M",           # Layer 5
]
VGG16 = [
          64,  64, 'M',      # Layer 1
         128, 128, 'M',      # Layer 2
    256, 256, 256, 'M',      # Layer 3
    512, 512, 512, 'M',      # Layer 4
    512, 512, 512, 'M',      # Layer 5
]
VGG19 = [
               64,  64, 'M', # Layer 1
              128, 128, 'M', # Layer 2
    256, 256, 256, 256, 'M', # Layer 3
    512, 512, 512, 512, 'M', # Layer 4
    512, 512, 512, 512, 'M', # Layer 5
]

if __name__ == "__main__":
    opt = Adam(lr=0.001)

    # test VGGNet16
    model = VGGNet(name = "VGGNet16", architecture = VGG16, input_shape=(224, 224, 3), classes = 1000)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    X_train = np.random.randn(1080, 224, 224, 3).astype('f')
    Y_train = np.random.randn(1080, 1000).astype('f')
    model.fit(X_train, Y_train, epochs = 1, batch_size = 32)