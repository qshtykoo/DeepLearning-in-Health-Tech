"""
ELEC- E8739 AI in Health Technologies. Exercise VII
Simulated dataset generation
Code by Zheng Zhao
Aalto University
zz@zabemon.com
Date: which I don't recall...
"""
import numpy as np
import matplotlib.pyplot as plt

import os


from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from tqdm import tqdm

# Two functions for generating dataset, choose one that you prefer.
def gen(n=10000, sigma=0.5):
    train = np.zeros(shape=(n, 2))
    x = np.linspace(-50, 50, n)
    y = np.sin(0.1*x)
    
    train[:,0] = x
    train[:,1] = np.random.normal(y, sigma)
        
    return train

def gen2(n=10000, sigma=10):
    train = np.zeros(shape=(n, 2))
    x = np.linspace(-50, 50, n)
    x = np.random.normal(x, sigma)

    train[:,0] = x
    train[:,1] = np.sin(0.1*x)
        
    return train

train_x = gen(n=20000, sigma=0.5)
np.save('sim_data.npy', train_x)

train_x = (train_x - np.mean(train_x))/(np.max(train_x) - np.mean(train_x)) #standardize the data -- very important

plt.scatter(train_x[:,0], train_x[:,1], s=1)
plt.show()

# The dimension of our random noise vector.
random_dim = 10
# You will use the Adam optimizer
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(64, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(32))
    
    generator.add(Dense(2, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(32, input_dim=2, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(64))
    
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator
  
def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def train(epochs=1, batch_size=128):
    # Get the training and testing data
    # Split the training data into batches of size 128
    batch_count = int(train_x.shape[0] / batch_size)

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)
    
    #d_loss = []
    #g_loss = []
    #d_loss_e = []
    #g_loss_e = []
    
    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = train_x[np.random.randint(0, train_x.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size) #we set v = D(x), y = 1 to obtain loss related to real images and v = D(G(x)), y = 0 to obtain loss related to fake images
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9    
            
            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)
            #d_loss_e.append(d_loss_1)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)   #we set v = D(G(z)), y = 1
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)
            #g_loss_e.append(g_loss_1)      
        #d_loss.append(np.mean(d_loss_e))
        #g_loss.append(np.mean(g_loss_e))
        #d_loss_e = []
        #g_loss_e = []
    return generator
            
if __name__ == '__main__':
    generator = train(20, 128)

	noise = np.random.normal(0, 1, size=[20000, random_dim])
	generated_data = generator.predict(noise)

	plt.scatter(generated_data[:,0], generated_data[:,1], s=1)
	plt.show()