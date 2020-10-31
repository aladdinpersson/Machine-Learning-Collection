# Tensorflow v.2.3.1

"""
Programmed by the-robot <https://github.com/the-robot>
"""

from tensorflow.keras.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def LeNet5(input_shape: typing.Tuple[int], classes: int = 1000) -> Model:
    """
    Implementation of the classic LeNet architecture.

    Arguments:
    input_shape -- shape of the images of the dataset
    classes     -- integer, number of classes

    Returns:
    model       -- a Model() instance in Keras

    Note:
    because I want to keep it original, I used tanh activation instead of ReLU activation.
    however based on newer papers, the rectified linear unit (ReLU) performed much faster than
    tanh activation.
    """

    # convert input shape into tensor
    X_input = Input(input_shape)

    # layer 1
    X = Conv2D(
        filters = 6,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "tanh",
        padding = "valid",
    )(X_input)
    X = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid")(X)

    # layer 2
    X = Conv2D(
        filters = 16,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "tanh",
        padding = "valid",
    )(X)
    X = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = "valid")(X)

    # layer 3
    X = Conv2D(
        filters = 120,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "tanh",
        padding = "valid",
    )(X)

    # layer 4
    X = Flatten()(X)
    X = Dense(units = 84, activation = "tanh")(X)

    # layer 5 (classification layer)
    X = Dense(units = classes, activation = "softmax")(X)

    model = Model(inputs = X_input, outputs = X, name = "LeNet5")
    return model