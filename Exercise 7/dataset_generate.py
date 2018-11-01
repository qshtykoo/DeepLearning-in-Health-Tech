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

plt.scatter(train_x[:,0], train_x[:,1], s=1)
plt.show()
