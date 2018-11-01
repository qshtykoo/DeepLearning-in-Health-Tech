import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#matplotlib inline

from scipy.stats import norm
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

K.clear_session()

mnist = fetch_mldata('MNIST original')
# normalize the data
X = mnist.data / 255.0
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

X_trainDf = pd.DataFrame(data = X_train)
y_trainDf = pd.DataFrame(data = y_train, columns = ['label'])
X_testDf = pd.DataFrame(data = X_test)
y_testDf = pd.DataFrame(data = y_test, columns = ['label'])
train_Df = pd.concat([X_trainDf, y_trainDf], axis = 1)
test_Df = pd.concat([X_testDf, y_testDf], axis = 1)
#combining training and testing dataset
total_Df = pd.concat([train_Df, test_Df], axis = 0, ignore_index = True) #the index does not have specific information

X_trainDf = X_trainDf.values.reshape(-1, 28, 28, 1)
X_testDf = X_testDf.values.reshape(-1, 28, 28, 1)

#show some samples from the training set
plt.figure(1)
plt.subplot(221)
plt.imshow(X_trainDf[13][:,:,0])

plt.subplot(222)
plt.imshow(X_trainDf[690][:,:,0])

plt.subplot(223)
plt.imshow(X_trainDf[2375][:,:,0])

plt.subplot(224)
plt.imshow(X_trainDf[42013][:,:,0])
plt.show()

# ---- Model Construction ------------ #

#-------------Encoder Network
img_shape = (28, 28, 1)    # for MNIST
batch_size = 16
latent_dim = 2  # Number of latent dimension parameters
# Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', 
                  activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
# need to know the shape of the network here for the decoder
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# Two outputs, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)

# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon

# sample vector from the latent distribution
z = layers.Lambda(sampling)([z_mu, z_log_sigma])


#---------------Decoder Network
# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:])

# Expand to 784 total pixels
x = layers.Dense(np.prod(shape_before_flattening[1:]), # 14 * 14 * 64
                 activation='relu')(decoder_input)

# reshape
x = layers.Reshape(shape_before_flattening[1:])(x)

# use Conv2DTranspose to reverse the conv layers from the encoder
x = layers.Conv2DTranspose(32, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)
x = layers.Conv2D(1, 3,
                  padding='same', 
                  activation='sigmoid')(x)

# decoder model statement
decoder = Model(decoder_input, x)

# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)

# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs) #Layer.add_loss()
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomVariationalLayer()([input_img, z_decoded])

# VAE model statement
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()
