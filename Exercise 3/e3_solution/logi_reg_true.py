'''
ELEC- E8739 AI in Health Technologies. Exercise III
Logistic Regression. The model is described in the exercise III.
Code by Zheng Zhao
Aalto University
zz@zabemon.com
Date: Sep 25, 2018
'''

# Import dependencies
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats

# Generate samples
n_samples = 1000 # Number of samples (datapoints)

w = 0.8	 # Ground truth weight
sig = 4  # Hyperparameter

# Define Sigmoid
def sigmoid(x, w):
    return 1 / (1 + np.exp(-w * x))

train_x = np.linspace(-12,12,n_samples)
train_y = np.random.binomial(1, sigmoid(train_x, w))

# Plot the data points
plt.figure()
plt.plot(train_x, sigmoid(train_x, w), color='orange', label='Ground truth sigmoid')
plt.scatter(train_x[0:-1:10], train_y[0:-1:10], s=6, marker='o', label='Some Samples') # Only show 100 samples for better looking
plt.legend()
plt.xlabel('w_true = %s' % (w))
plt.title('Simulated Logistic Regression Data')
plt.show()

# w_est = train_y @ train_x @ np.linalg.inv(train_x.transpose() @ train_x)
# w_post_est = train_y @ train_x @ np.linalg.inv(train_x.transpose() @ train_x + 2*np.eye(2))

###
# Here using Tensorflow for logistic regression by optimizing MLE
###

# Parameters for training
learning_r = 6e-2 # Try different learning rate, and find out what will happen
epochs = 500

# Some reshape of data
train_x = train_x.reshape(n_samples, 1)
train_y = train_y.reshape(n_samples, 1)
# Here defines Tensorflow Graph
x = tf.placeholder(tf.float64, shape=(None, 1))  # Here defines nodes for accepting input data
y = tf.placeholder(tf.float64, shape=(None, 1))

w_est = tf.Variable(1, name="weight", dtype=tf.float64) # Try different initial value

p = tf.divide(1, 1 + tf.exp(-w_est * x))

# Here defines the loss. Maximizing MLE leads to minimizing cross entropy
loss = -1 * tf.reduce_sum(y * tf.log(p) + (1 - y) * tf.log(1 - p)) / n_samples

# Here defines Optimization
optimizer = tf.train.GradientDescentOptimizer(learning_r).minimize(loss)

# Initialization
init = tf.global_variables_initializer()

# History
history = np.zeros(shape=(epochs,))

# Run session using context manager
with tf.Session() as sess:
    sess.run(init)
  
    for e in range(epochs):
        sess.run(optimizer, feed_dict={x: train_x, y: train_y})
        history[e] = sess.run(loss, feed_dict={x: train_x, y: train_y})
        print("loss: %f, w_est: %s" % (history[e], sess.run(w_est)))

    # Print and plot results
    w_est = sess.run(w_est)
    print('Estimated weight: %s' % (w_est))
    print('True weight: %s' % (w))
    plt.figure()
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.xlabel('Loss')
    plt.figure()
    plt.plot(train_x, sigmoid(train_x, w), color='orange', label='Ground truth sigmoid')
    plt.plot(train_x, sigmoid(train_x, w_est), 'b--', label='Estimated sigmoid')
    plt.legend()
    plt.title('Logistic Regression Result')
    plt.show()
