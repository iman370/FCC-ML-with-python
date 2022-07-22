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
# Why? The weights and bias will be between 0 and 1, so at the start, they will be insignificant (if we didn't pre-process the data)
# The NN will have to work much harder to update the weights and biases (much harder if there's more than 255 values)
train_images = train_images / 255.0

test_images = test_images / 255.0

# Creating the model - basically creating the architecutre of the NN
model = keras.Sequential([ # Sequential is the basic neural network that goes through each node sequentially
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3) - the 10 means that we have 10 output neurons. 1 node for each class name
]) # softmax makes sure that all of the neurons add up to 1 and all of them have a value between 0 and 1

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

print('Test accuracy:', test_acc)

# Making predictions
predictions = model.predict(test_images)

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  print("Excpected: " + label)
  print("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)