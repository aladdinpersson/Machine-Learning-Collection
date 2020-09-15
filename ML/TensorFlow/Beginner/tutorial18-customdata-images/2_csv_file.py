import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

directory = "data/mnist_images_csv/"
df = pd.read_csv(directory + "train.csv")

file_paths = df["file_name"].values
labels = df["label"].values
ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))


def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    return image, label


def augment(image, label):
    # data augmentation here
    return image, label


ds_train = ds_train.map(read_image).map(augment).batch(2)

for epoch in range(10):
    for x, y in ds_train:
        # train here
        pass

model = keras.Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=2)
