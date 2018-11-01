'''
ELEC- E8739 AI in Health Technologies. Exercise III
Linear Regression. The model is described in the exercise III.
Here the general framework of codes have been given. You need to fill in the ??? part.
Code by Zheng Zhao
Aalto University
zz@zabemon.com
Date: Sep 19, 2018
'''

# Import dependencies
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats

# Generate simulated data
n_samples = 1000  # Number of samples (datapoints)

w = 0.8
sig = 4

def sigmoid(x, y):
    return 1 / (1 + np.exp(-x*y))

train_x = np.linspace(-12, 12, n_samples).reshape(n_samples,1)
train_y = np.random.binomial(1, sigmoid(train_x,w)) #some bernoulli distribution
# Plot the data points
plt.figure()
plt.scatter(train_x, train_y, label='Samples', s=3)
plt.plot(train_x, sigmoid(train_x,w), label='Sigmoid Function', color='red')
plt.title('Simulated Data')
plt.show()

###
# Here using Tensorflow for linear regression by optimizing MLE
###

learning_r = 6e-2  # Try different learning rate, and find out what will happen
epochs = 500  # Iteration

# Here defines Tensorflow Graph
x = tf.placeholder("float", shape=(None, 1))  # Here defines nodes for accepting input data
y = tf.placeholder("float", shape=(None, 1))

w_est = tf.Variable(1.0, name="weight")  # Here defines estimated weight to be optimized. Try different initial value

# in this exercise the logistic function is sigmoid function
pred = tf.nn.sigmoid(x * w_est)
#pred = tf.divide(1, 1 + tf.exp(-w_est * x))

# Here defines the logistic loss
# tf.reduce_sum() Computes the sum of elements across dimensions of a tensor
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred) + (1-y) * tf.log(1-pred), reduction_indices=1)) #for SGD, the initial loss should not be too large
#loss = -1 * tf.reduce_sum(y * tf.log(pred) + (1 - y) * tf.log(1 - pred)) / n_samples

# Here defines SGD optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_r).minimize(loss)

# Initialization
init = tf.global_variables_initializer()

# History of loss
history = np.zeros(shape=(epochs,))

# Run session using context manager
with tf.Session() as sess:
    sess.run(init)

    for e in range(epochs):
        sess.run(optimizer, feed_dict={x: train_x, y: train_y})
        history[e] = sess.run(loss, feed_dict={x: train_x, y: train_y})
        print("loss: %f, w_est: %s" % (history[e], sess.run(w_est)))

    # Print and plot results
    print("Optimization Finished!")
    print('Estimated weight: %s' % (sess.run(w_est)))
    print('True weight: %s' % (w))
    plt.figure()
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
