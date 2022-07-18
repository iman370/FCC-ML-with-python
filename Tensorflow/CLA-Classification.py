# For this project, we're going to identify different types of flowers
# Classification is the process of separating data points into different classes.

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

# Lets define some constants to help us later on
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# Get rid of the species column
train_y = train.pop('Species')
test_y = test.pop('Species')

# Create input function
# This time, we're creating the input function straight up instead of making a function that creates the input function
# Also, notice how there is no epoch
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys(): # All of our columns are numeric, hence we only need 1 loop
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#print(my_feature_columns)