# Tensorflow v.2.3.1

"""
Programmed by the-robot <https://github.com/the-robot>
"""

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def VGGNet(
    name: str,
    architecture: typing.List[ typing.Union[int, str] ],
    input_shape: typing.Tuple[int],
    classes: int = 1000
) -> Model:
    """
    Implementation of the VGGNet architecture.

    Arguments:
    name         -- name of the architecture
    architecture -- number of output channel per convolution layers in VGGNet
    input_shape  -- shape of the images of the dataset
    classes      -- integer, number of classes

    Returns:
    model        -- a Model() instance in Keras
    """

    # convert input shape into tensor
    X_input = Input(input_shape)

    # make convolution layers
    X = make_conv_layer(X_input, architecture)

    # flatten the output and make fully connected layers
    X = Flatten()(X)
    X = make_dense_layer(X, 4096)
    X = make_dense_layer(X, 4096)

    # classification layer
    X = Dense(units = classes, activation = "softmax")(X)

    model = Model(inputs = X_input, outputs = X, name = name)
    return model

def make_conv_layer(
    X: tf.Tensor,
    architecture: typing.List[ typing.Union[int, str] ],
    activation: str = 'relu'
) -> tf.Tensor:
    """
    Method to create convolution layers for VGGNet.
    In VGGNet
        - Kernal is always 3x3 for conv-layer with padding 1 and stride 1.
        - 2x2 kernel for max pooling with stride of 2.

    Arguments:
    X            -- input tensor
    architecture -- number of output channel per convolution layers in VGGNet
    activation   -- type of activation method

    Returns:
    X           -- output tensor
    """

    for output in architecture:

        # convolution layer
        if type(output) == int:
            out_channels = output

            X = Conv2D(
                filters = out_channels,
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = "same"
            )(X)
            X = BatchNormalization()(X)
            X = Activation(activation)(X)

            # relu activation is added (by default activation) so that all the
            # negative values are not passed to the next layer

        # max-pooling layer
        else:
            X = MaxPooling2D(
                pool_size = (2, 2),
                strides = (2, 2)
            )(X)

    return X

def make_dense_layer(X: tf.Tensor, output_units: int, dropout = 0.5, activation = 'relu') -> tf.Tensor:
    """
    Method to create dense layer for VGGNet.

    Arguments:
    X            -- input tensor
    output_units -- output tensor size
    dropout      -- dropout value for regularization
    activation   -- type of activation method

    Returns:
    X            -- input tensor
    """

    X = Dense(units = output_units)(X)
    X = BatchNormalization()(X)
    X = Activation(activation)(X)
    X = Dropout(dropout)(X)

    return X