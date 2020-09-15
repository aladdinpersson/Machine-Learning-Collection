import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Initialization
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
print(x)

x = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2, 3))
print(x)

x = tf.eye(3)
print(x)

x = tf.ones((4, 3))
print(x)

x = tf.zeros((3, 2, 5))
print(x)

x = tf.random.uniform((2, 2), minval=0, maxval=1)
print(x)

x = tf.random.normal((3, 3), mean=0, stddev=1)
print(tf.cast(x, dtype=tf.float64))
# tf.float (16,32,64), tf.int (8, 16, 32, 64), tf.bool

x = tf.range(9)
x = tf.range(start=0, limit=10, delta=2)
print(x)

# Math
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

z = tf.add(x, y)
z = x + y

z = tf.subtract(x, y)
z = x - y

z = tf.divide(x, y)
z = x / y

z = tf.multiply(x, y)
z = x * y

z = tf.tensordot(x, y, axes=1)

z = x ** 5

x = tf.random.normal((2, 3))
y = tf.random.normal((3, 2))
z = tf.matmul(x, y)
z = x @ y

x = tf.random.normal((2, 2))

# Indexing
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[:])
print(x[1:])
print(x[1:3])
print(x[::2])
print(x[::-1])

indices = tf.constant([0, 3])
x_indices = tf.gather(x, indices)

x = tf.constant([[1, 2], [3, 4], [5, 6]])

print(x[0, :])
print(x[0:2, :])

# Reshaping
x = tf.range(9)

x = tf.reshape(x, (3, 3))

x = tf.transpose(x, perm=[1, 0])
