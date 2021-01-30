import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers

# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

writer = tf.summary.create_file_writer("logs/graph_vis")


@tf.function
def my_func(x, y):
    return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

tf.summary.trace_on(graph=True, profiler=True)
out = my_func(x, y)

with writer.as_default():
    tf.summary.trace_export(
        name="function_trace", step=0, profiler_outdir="logs\\graph_vis\\"
    )
