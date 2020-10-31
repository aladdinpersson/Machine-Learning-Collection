# Tensorflow v.2.3.1

"""
Programmed by the-robot <https://github.com/the-robot>
"""

from block import (
    auxiliary_block,
    convolution_block,
    inception_block,
)

from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Dropout,
    Input,
    MaxPooling2D,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def GoogLeNet(input_shape: typing.Tuple[int] = (224, 224, 3), classes: int = 1000) -> Model:
    """
    Implementation of the popular GoogLeNet aka Inception v1 architecture.
    Refer to the original paper, page 6 - table 1 for inception block filter sizes.
    Arguments:
    input_shape -- shape of the images of the dataset
    classes     -- number of classes for classification
    Returns:
    model       -- a Model() instance in Keras
    """

    # convert input shape into tensor
    X_input = Input(input_shape)

    # NOTE: auxiliary layers are only used in trainig phase to improve performance
    #       because they act as regularization and prevent vanishing gradient problem
    auxiliary1 = None # to store auxiliary layers classification value
    auxiliary2 = None

    # layer 1 (convolution block)
    X = convolution_block(
        X = X_input,
        filters = 64,
        kernel_size = 7,
        stride = 2,
        padding = "same",
    )

    # layer 2 (max pool)
    X = MaxPooling2D(
        pool_size = (3, 3),
        padding = "same",
        strides = (2, 2),
    )(X)

    # layer 3 (convolution block)
    # 1x1 reduce
    X = convolution_block(
        X,
        filters = 64,
        kernel_size = 1,
        stride = 1,
        padding = "same",
    )
    X = convolution_block(
        X,
        filters = 192,
        kernel_size = 3,
        stride = 1,
        padding = "same",
    )

    # layer 4 (max pool)
    X = MaxPooling2D(
        pool_size = (3, 3),
        padding = "same",
        strides = (2, 2),
    )(X)

    # layer 5 (inception 3a)
    X = inception_block(
        X,
        filters_1x1 = 64,
        filters_3x3_reduce = 96,
        filters_3x3 = 128,
        filters_5x5_reduce = 16,
        filters_5x5 = 32,
        pool_size = 32,
    )

    # layer 6 (inception 3b)
    X = inception_block(
        X,
        filters_1x1 = 128,
        filters_3x3_reduce = 128,
        filters_3x3 = 192,
        filters_5x5_reduce = 32,
        filters_5x5 = 96,
        pool_size = 64,
    )

    # layer 7 (max pool)
    X = MaxPooling2D(
        pool_size = (3, 3),
        padding = "same",
        strides = (2, 2),
    )(X)

    # layer 8 (inception 4a)
    X = inception_block(
        X,
        filters_1x1 = 192,
        filters_3x3_reduce = 96,
        filters_3x3 = 208,
        filters_5x5_reduce = 16,
        filters_5x5 = 48,
        pool_size = 64,
    )

    # First Auxiliary Softmax Classifier
    auxiliary1 = auxiliary_block(X, classes = classes)

    # layer 9 (inception 4b)
    X = inception_block(
        X,
        filters_1x1 = 160,
        filters_3x3_reduce = 112,
        filters_3x3 = 224,
        filters_5x5_reduce = 24,
        filters_5x5 = 64,
        pool_size = 64,
    )

    # layer 10 (inception 4c)
    X = inception_block(
        X,
        filters_1x1 = 128,
        filters_3x3_reduce = 128,
        filters_3x3 = 256,
        filters_5x5_reduce = 24,
        filters_5x5 = 64,
        pool_size = 64,
    )

    # layer 11 (inception 4d)
    X = inception_block(
        X,
        filters_1x1 = 112,
        filters_3x3_reduce = 144,
        filters_3x3 = 288,
        filters_5x5_reduce = 32,
        filters_5x5 = 64,
        pool_size = 64,
    )

    # Second Auxiliary Softmax Classifier
    auxiliary2 = auxiliary_block(X, classes = classes)

    # layer 12 (inception 4e)
    X = inception_block(
        X,
        filters_1x1 = 256,
        filters_3x3_reduce = 160,
        filters_3x3 = 320,
        filters_5x5_reduce = 32,
        filters_5x5 = 128,
        pool_size = 128,
    )

    # layer 13 (max pool)
    X = MaxPooling2D(
        pool_size = (3, 3),
        padding = "same",
        strides = (2, 2),
    )(X)

    # layer 14 (inception 5a)
    X = inception_block(
        X,
        filters_1x1 = 256,
        filters_3x3_reduce = 160,
        filters_3x3 = 320,
        filters_5x5_reduce = 32,
        filters_5x5 = 128,
        pool_size = 128,
    )

    # layer 15 (inception 5b)
    X = inception_block(
        X,
        filters_1x1 = 384,
        filters_3x3_reduce = 192,
        filters_3x3 = 384,
        filters_5x5_reduce = 48,
        filters_5x5 = 128,
        pool_size = 128,
    )

    # layer 16 (average pool)
    X = AveragePooling2D(
        pool_size = (7, 7),
        padding = "same",
        strides = (1, 1),
    )(X)

    # layer 17 (dropout 40%)
    X = Dropout(rate = 0.4)(X)

    # layer 18 (fully-connected layer with softmax activation)
    X = Dense(units = classes, activation='softmax')(X)

    model = Model(X_input, outputs = [X, auxiliary1, auxiliary2], name='GoogLeNet/Inception-v1')
    return model
