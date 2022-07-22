# We're going to create a neural network that identifies clothing items

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# We'll use 60k imagines for training and 10k images for testing
fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

# This should return [60000,28,28]
# 60k images, each which are 28x28 pixels
#train_images.shape

# This should have a value between 0 and 255 (pixel colour as it is grey-scale)
#train_images[0,23,23]  # let's have a look at one pixel

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Looking at some images
#plt.figure()
#plt.imshow(train_images[4])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Data pre-processing
# We will squish all of our values so they range between 0 and 1
train_images = train_images / 255.0

test_images = test_images / 255.0