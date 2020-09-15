import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers

# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


def augment(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label


# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

class_names = [
    "Airplane",
    "Autmobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


def train_model_one_epoch(hparams):
    units = hparams[HP_NUM_UNITS]
    drop_rate = hparams[HP_DROPOUT]
    learning_rate = hparams[HP_LR]

    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model = keras.Sequential(
        [
            layers.Input((32, 32, 3)),
            layers.Conv2D(8, 3, padding="same", activation="relu"),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(units, activation="relu"),
            layers.Dropout(drop_rate),
            layers.Dense(10),
        ]
    )

    for batch_idx, (x, y) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y, y_pred)

    # write to TB
    run_dir = (
        "logs/train/"
        + str(units)
        + "units_"
        + str(drop_rate)
        + "dropout_"
        + str(learning_rate)
        + "learning_rate"
    )

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = acc_metric.result()
        tf.summary.scalar("accuracy", accuracy, step=1)

    acc_metric.reset_states()


loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
HP_NUM_UNITS = hp.HParam("num units", hp.Discrete([32, 64, 128]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.1, 0.2, 0.3, 0.5]))
HP_LR = hp.HParam("learning_rate", hp.Discrete([1e-3, 1e-4, 1e-5]))

for lr in HP_LR.domain.values:
    for units in HP_NUM_UNITS.domain.values:
        for rate in HP_DROPOUT.domain.values:
            hparams = {
                HP_LR: lr,
                HP_NUM_UNITS: units,
                HP_DROPOUT: rate,
            }

            train_model_one_epoch(hparams)
