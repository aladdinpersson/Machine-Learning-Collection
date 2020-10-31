# Tensorflow v.2.3.1

"""
Programmed by the-robot <https://github.com/the-robot>
"""

from block import block

from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def ResNet(name: str, layers: typing.List[int], input_shape: typing.Tuple[int] = (64, 64, 3), classes: int = 6) -> Model:
    """
    Implementation of the popular ResNet architecture.

    Arguments:
    name        -- name of the architecture
    layers      -- number of blocks per layer
    input_shape -- shape of the images of the dataset
    classes     -- integer, number of classes

    Returns:
    model       -- a Model() instance in Keras


    Model Architecture:
    Resnet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1
            -> CONVBLOCK -> IDBLOCK * 2         // conv2_x
            -> CONVBLOCK -> IDBLOCK * 3         // conv3_x
            -> CONVBLOCK -> IDBLOCK * 5         // conv4_x
            -> CONVBLOCK -> IDBLOCK * 2         // conv5_x
            -> AVGPOOL
            -> TOPLAYER

    Resnet101:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1
            -> CONVBLOCK -> IDBLOCK * 2         // conv2_x
            -> CONVBLOCK -> IDBLOCK * 3         // conv3_x
            -> CONVBLOCK -> IDBLOCK * 22        // conv4_x
            -> CONVBLOCK -> IDBLOCK * 2         // conv5_x
            -> AVGPOOL
            -> TOPLAYER

    Resnet152:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1
            -> CONVBLOCK -> IDBLOCK * 2         // conv2_x
            -> CONVBLOCK -> IDBLOCK * 7         // conv3_x
            -> CONVBLOCK -> IDBLOCK * 35        // conv4_x
            -> CONVBLOCK -> IDBLOCK * 2         // conv5_x
            -> AVGPOOL
            -> TOPLAYER
    """

    # get layers (layer1 is always the same so no need to provide)
    layer2, layer3, layer4, layer5 = layers

    # convert input shape into tensor
    X_input = Input(input_shape)

    # zero-padding
    X = ZeroPadding2D((3, 3))(X_input)

    # conv1
    X = Conv2D(
        filters = 64,
        kernel_size = (7, 7),
        strides = (2, 2),
        name = "conv1",
        kernel_initializer = "glorot_uniform",
    )(X)
    X = BatchNormalization(axis = 3, name = "bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)

    # conv2_x
    X = make_layer(X, layers = layer2, kernel_size = 3, filters = [64, 64, 256], stride = 1, stage_no = 2)

    # conv3_x
    X = make_layer(X, layers = layer3, kernel_size = 3, filters = [128, 128, 512], stride = 2, stage_no = 3)

    # conv4_x
    X = make_layer(X, layers = layer4, kernel_size = 3, filters = [256, 256, 1024], stride = 2, stage_no = 4)

    # conv5_x
    X = make_layer(X, layers = layer5, kernel_size = 3, filters = [512, 512, 2048], stride = 1, stage_no = 5)

    # average pooling
    X = AveragePooling2D((2, 2), name = "avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dense(
        classes,
        activation = "softmax",
        name="fc" + str(classes),
        kernel_initializer = "glorot_uniform"
    )(X)

    model = Model(inputs = X_input, outputs = X, name = name)
    return model

def make_layer(X: tf.Tensor, layers: int, kernel_size: int, filters: typing.List[int], stride: int, stage_no: int) -> tf.Tensor:
    """
    Method to create one conv-identity layer for ResNet.

    Arguments:
    X           -- input tensor
    layers      -- number of blocks per layer
    kernel_size -- size of the kernel for the block
    filters     -- number of filters/channels
    stride      -- number of stride for downsampling the input
    stage_no    -- stage number just to name the layer

    Returns:
    X           -- output tensor
    """

    # create convolution block
    X = block(
        X,
        kernel_size = kernel_size,
        filters = filters,
        stage_no = stage_no,
        block_name = "a",
        is_conv_layer = True,
        stride = stride
    )

    # create identity block
    block_name_ordinal = ord("b")
    for _ in range(layers - 1):
        X = block(
            X,
            kernel_size = kernel_size,
            filters =  filters,
            stage_no = stage_no,
            block_name = chr(block_name_ordinal)
        )
        block_name_ordinal += 1

    return X