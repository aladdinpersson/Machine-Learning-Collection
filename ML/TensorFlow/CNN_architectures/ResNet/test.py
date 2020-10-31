# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from resnet import ResNet

import numpy as np

if __name__ == "__main__":
    # test ResNet50
    model = ResNet(name = "Resnet50", layers = [3, 4, 6, 3], input_shape = (64, 64, 3), classes = 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_train = np.random.randn(1080, 64, 64, 3).astype('f')
    Y_train = np.random.randn(1080, 6).astype('f')
    model.fit(X_train, Y_train, epochs = 1, batch_size = 32)

    X_test = np.random.rand(10, 64, 64, 3).astype('f')
    Y_test = np.random.rand(10, 6).astype('f')
    preds = model.evaluate(X_test, Y_test)

    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))