import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ================================================ #
#                  Pretrained-Model                #
# ================================================ #

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = keras.models.load_model("pretrained")

# Freeze all model layer weights
model.trainable = False

# Can also set trainable for specific layers
for layer in model.layers:
    # assert should be true because of one-liner above
    assert layer.trainable == False
    layer.trainable = False

print(model.summary())  # for finding base input and output
base_inputs = model.layers[0].input
base_output = model.layers[-2].output
output = layers.Dense(10)(base_output)
new_model = keras.Model(base_inputs, output)

# This model is actually identical to model we
# loaded (this is just for demonstration and
# and not something you would do in practice).
print(new_model.summary())

# As usual we do compile and fit, this time on new_model
new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

new_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)

# =================================================== #
#                Pretrained Keras Model               #
# =================================================== #

# Random data for demonstration (3 examples w. 3 classes)
x = tf.random.normal(shape=(3, 299, 299, 3))
y = tf.constant([0, 1, 2])

model = keras.applications.InceptionV3(include_top=True)
print(model.summary())

# for input you can also do model.input,
# then for base_outputs you can obviously
# choose other than simply removing the last one :)
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
classifier = layers.Dense(3)(base_outputs)
new_model = keras.Model(inputs=base_inputs, outputs=classifier)
new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print(new_model.summary())
new_model.fit(x, y, epochs=15, verbose=2)

# ================================================= #
#                Pretrained Hub Model               #
# ================================================= #

# Random data for demonstration (3 examples w. 3 classes)
x = tf.random.normal(shape=(3, 299, 299, 3))
y = tf.constant([0, 1, 2])

url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"

base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
model = keras.Sequential(
    [
        base_model,
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x, y, batch_size=32, epochs=15, verbose=2)
