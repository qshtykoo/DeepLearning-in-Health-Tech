import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras_sequential_ascii import sequential_model_to_ascii_printout

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp

# Loading the CIFAR-10 datasets
from keras.datasets import cifar10

# Declare variables
batch_size = 128 #the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you’ll need
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10  # number of cifar-10 data set classes
epochs = 5  # repeat 100 times
# one forward pass and one backward pass of all the training examples

#load cifar10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train - training data(images), y_train - labels(digits)
class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#print figure with 10 random images from each class
fig = plt.figure(figsize=(8, 3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:] == i)[0]
    features_idx = x_train[idx, ::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num, ::], (1, 2, 0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# convert and pre-processing
# perform normalization on the data
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

def base_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:])) #Convolutional input layer, 32 feature maps with a size of 3×3 filter
    model.add(Activation('relu')) #a rectifier activation function
    model.add(Conv2D(32, (3, 3))) #Convolutional input layer, 32 feature maps with a size of 3×3 filter
    model.add(Activation('relu')) #a rectifier activation function
    model.add(MaxPooling2D(pool_size=(2, 2))) #Max Pooling layer with size 2×2
    model.add(Dropout(0.25)) #Dropout set to 25%

    model.add(Conv2D(64, (3, 3), padding='same')) #Convolutional input layer, 64 feature maps with a size of 3×3 filter
    model.add(Activation('relu')) #a rectifier activation function
    model.add(Conv2D(64, (3, 3))) #Convolutional input layer, 64 feature maps with a size of 3×3 filter
    model.add(Activation('relu')) #a rectifier activation function
    model.add(MaxPooling2D(pool_size=(2, 2))) #Max Pooling layer with size 2×2
    model.add(Dropout(0.25)) #Dropout set to 25%

    model.add(Flatten()) #flatten layer
    model.add(Dense(512)) #Fully connected layer with 512 units
    model.add(Activation('relu')) #and a rectifier activation function
    model.add(Dropout(0.5)) #dropout set to 50%
    model.add(Dense(num_classes)) #Fully connected output layer with 10 units
    model.add(Activation('softmax')) ## and a softmax activation function

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    return model

cnn_n = base_model()
cnn_n.summary()

# Fit model
cnn = cnn_n.fit(x_train / 255.0, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test / 255.0, y_test), shuffle=True)

# Plots for training and testing process: loss and accuracy
plt.figure(0)
plt.plot(cnn.history['acc'], 'r')
plt.plot(cnn.history['val_acc'], 'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])

plt.figure(1)
plt.plot(cnn.history['loss'], 'r')
plt.plot(cnn.history['val_loss'], 'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])

plt.show()
