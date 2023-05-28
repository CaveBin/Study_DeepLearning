import tensorflow as tf
from fit import Fit
from keras.datasets import mnist

from naiveDense import NaiveDense
from naiveSequential import NaiveSequential
#import m_model

model = NaiveSequential([
    NaiveDense(input_size=28*28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])

assert len(model.weights) == 4

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

Fit(model, train_images, train_labels, epochs=10, batch_size=128)