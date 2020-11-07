  # Tensorflow v.2.3.1

"""
Programmed by the-robot <https://github.com/the-robot>
"""

from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    concatenate,
)
import tensorflow as tf
import typing

@tf.function
def convolution_block(
    X: tf.Tensor,
    filters: int,
    kernel_size: int,
    stride: int = 1,
    padding: str = 'valid',
) -> tf.Tensor:
    """
    Convolution block for GoogLeNet.
    Arguments:
    X           -- input tensor of shape (m, H, W, filters)
    filters      -- defining the number of filters in the CONV layers
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    stride      -- integer specifying the stride to be used
    padding     -- padding type, same or valid. Default is valid
    Returns:
    X           -- output of the identity block, tensor of shape (H, W, filters)
    """

    X = Conv2D(
        filters = filters,
        kernel_size = (kernel_size, kernel_size),
        strides = (stride, stride),
        padding = padding,
    )(X)
    # batch normalization is not in original paper because it was not invented at that time
    # however I am using it here because it will improve the performance
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    return X

@tf.function
def inception_block(
    X: tf.Tensor,
    filters_1x1: int,
    filters_3x3_reduce: int,
    filters_3x3: int,
    filters_5x5_reduce: int,
    filters_5x5: int,
    pool_size: int,
) -> tf.Tensor:
    """
    Inception block for GoogLeNet.
    Arguments:
    X                  -- input tensor of shape (m, H, W, filters)
    filters_1x1        -- number of filters for (1x1 conv) in first branch 
    filters_3x3_reduce -- number of filters for (1x1 conv) dimensionality reduction before (3x3 conv) in second branch
    filters_3x3        -- number of filters for (3x3 conv) in second branch
    filters_5x5_reduce -- number of filters for (1x1 conv) dimensionality reduction before (5x5 conv) in third branch
    filters_5x5        -- number of filters for (5x5 conv) in third branch
    pool_size          -- number of filters for (1x1 conv) after 3x3 max pooling in fourth branch 
    Returns:
    X                  -- output of the identity block, tensor of shape (H, W, filters)
    """

    # first branch
    conv_1x1 = convolution_block(
        X,
        filters = filters_1x1,
        kernel_size = 1,
        padding = "same"
    )

    # second branch
    conv_3x3 = convolution_block(
        X,
        filters = filters_3x3_reduce,
        kernel_size = 1,
        padding = "same"
    )
    conv_3x3 = convolution_block(
        conv_3x3,
        filters = filters_3x3,
        kernel_size = 3,
        padding = "same"
    )

    # third branch
    conv_5x5 = convolution_block(
        X,
        filters = filters_5x5_reduce,
        kernel_size = 1,
        padding = "same"
    )
    conv_5x5 = convolution_block(
        conv_5x5,
        filters = filters_5x5,
        kernel_size = 5,
        padding = "same"
    )

    # fourth branch
    pool_projection = MaxPooling2D(
        pool_size = (2, 2),
        strides = (1, 1),
        padding = "same",
    )(X)
    pool_projection = convolution_block(
        pool_projection,
        filters = pool_size,
        kernel_size = 1,
        padding = "same"
    )

    # concat by channel/filter
    return concatenate(inputs = [conv_1x1, conv_3x3, conv_5x5, pool_projection], axis = 3)

@tf.function
def auxiliary_block(
    X: tf.Tensor,
    classes: int,
) -> tf.Tensor:
    """
    Auxiliary block for GoogLeNet.
    Refer to the original paper, page 8 for the auxiliary layer specification.
    Arguments:
    X       -- input tensor of shape (m, H, W, filters)
    classes -- number of classes for classification
    Return:
    X       -- output of the identity block, tensor of shape (H, W, filters)
    """

    X = AveragePooling2D(
        pool_size = (5, 5),
        padding = "same",
        strides = (3, 3),
    )(X)
    X = convolution_block(
        X,
        filters = 128,
        kernel_size = 1,
        stride = 1,
        padding = "same",
    )
    X = Flatten()(X)
    X = Dense(units = 1024, activation = "relu")(X)
    X = Dropout(rate = 0.7)(X)
    X = Dense(units = classes)(X)
    X = Activation("softmax")(X)

    return X