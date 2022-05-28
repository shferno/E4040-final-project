"""
Self implementation of training process with details. 
"""


from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from model import fishnet99, fishnet55

from imgprocess import make_dataset

tf.test.gpu_device_name()

########################## Load Data #############################
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import mnist

# Load the raw data.
train, test = cifar10.load_data()
train, test = mnist.load_data()
train, test = cifar100.load_data()

#X_train_raw, y_train = train
#X_test_raw, y_test = test

######################## Data organizations #######################
# Train data: 1900 samples from original train set: 1~4,500
# Validation data: 100 samples from original train set: 4,500~5,000
# Test data: 500 samples from original test set: 1~500
num_training = 4500
num_validation = 500
num_test = 500

X_train_subset = X_train_raw[:(num_training+num_validation), :]
y_train_subset = y_train[:(num_training+num_validation)]

X_train_sub = X_train_subset[:num_training, :]
y_train_sub = y_train_subset[:num_training]

X_val_sub = X_train_subset[-num_validation:, :]
y_val_sub = y_train_subset[-num_validation:]

X_test_sub = X_test_raw[:num_test, :]
y_test_sub = y_test[:num_test]

# Check Data Shape
print(X_train_sub.shape, X_val_sub.shape, X_test_sub.shape)
print('Train data shape: ', X_train_sub.shape)
print('Train labels shape: ', y_train_sub.shape)
print('Validation data shape: ', X_val_sub.shape)
print('Validation labels shape: ', y_val_sub.shape)
print('Test data shape: ', X_test_sub.shape)
print('Test labels shape: ', y_test_sub.shape)

## Data Visualization
# cifar10: [0]airplane, [1]automobile, [2]bird, [3]cat, [4]deer, [5]dog, [6]frog, [7]horse, [8]ship, [9]truck
# Preview random image
from imgprocess import plot

#plot(X_test_sub, y_test_sub)

######################### Data Preprocess ##########################
# Data Resize
from imgprocess import resize
resize(x, shape = (,,))

shape = (56,56,1)
shape = (224,224,3)
X_train_resize = resize(X_train_sub, shape)
X_val_resize = resize(X_val_sub, shape)
X_test_resize = resize(X_test_sub, shape)

# Check Data Shape
print('Train data shape: ', X_train_resize.shape)
print('Validation data shape: ', X_val_resize.shape)
print('Test data shape: ', X_test_resize.shape)

# Preprocessing: subtract the mean value across every dimension for training data
mean_image = np.mean(X_train_resize, axis=0)

X_train = X_train_resize.astype(np.float32) - mean_image.astype(np.float32)
X_val = X_val_resize.astype(np.float32) - mean_image
X_test = X_test_resize.astype(np.float32) - mean_image

y_train = y_train_sub
y_val = y_val_sub
y_test = y_test_sub

# Check Data Shape
print(X_train.shape, X_val.shape, X_test.shape)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

########################## import model ##########################
## Summary
#tf.keras.backend.clear_session()
def fish():
    model = fishnet55()
    model.summary(expand_nested = True)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(1e-2, momentum=0.9),
        #optimizer=tf.keras.optimizers.Adam(1e-2),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(
        x=X_train, y=y_train, verbose=1,
        batch_size=32, epochs=20,
        validation_data=(X_val, y_val)
    )

def mlp():
    mlp = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(224,224,3)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(100)
    ])
    mlp.compile(
        optimizer = tf.keras.optimizers.SGD(0.01),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    mlp.fit(
        x=X_train, y=y_train, verbose=1,
        batch_size=64, epochs=50,
        validation_data=(X_val, y_val)
    )

def fish55():
    batch_size = 64
    train, test = cifar10.load_data()
    train_set, val_set = make_dataset(train, test, batch_size)
    model = fishnet55(num_cls=10, input_shape=(56,56,3))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(1e-3, momentum=0.9), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy']
    )
    model.fit(
        x=train_set, verbose=1, 
        batch_size=batch_size, epochs=30, 
        validation_data=val_set
    )


if __name__ == "__main__":
    fish55()
    #mlp()
