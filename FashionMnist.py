from __future__ import print_function

import tensorflow.keras

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

# Suppress warn and info msg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Number of classes - do not change unless the data changes
num_classes = 10

# Sizes of batch and # of epochs of data
batch_size = 128
epochs = 24

# input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled, and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# Reshape the data
# deal with format issues between different backends.
# Some put the # of channels in the image before the width and height params
if K.image_data_format() == 'channels_first':
    # 1 - refers to the color channels; this is grayscale
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Scale the Pixel intensity (ranges from 0 to 255)
# Type convert and scale the test and training data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vecors to binary class matrices.
# One-hot encoding
# 3 => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)


# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Define the compile to minimize categorical loss, use ada delta optimized
# and optimize to maximizing accuracy
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the model and test/validate the mode with the test data after each epoch (cycle)
# through the training data;
# Return history of loss and accuracy for each epoch
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(x_test, y_test))

# Evaluate the model with the test data to get the scores on "real" data
score = model.evaluate(x_test, y_test, verbose=1)
print("[+] Total loss:\t", score[0])
print("[+] Total accuracy:\t", score[1])


# Plot the data to see relationships
import numpy as np
import matplotlib.pyplot as plt
epoch_list = list(range(1, len(hist.history['acc']) + 1))    # Values for x axis [1, 2, ... ,# of epochs
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()


