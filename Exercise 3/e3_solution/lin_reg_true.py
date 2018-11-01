'''
ELEC- E8739 AI in Health Technologies. Exercise III
Linear Regression. The model is described in the exercise III.
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

# Generate samples
n_samples = 1000 # Number of samples (datapoints)

w = np.array([2,5]).reshape(1,2)	 # Ground truth weight
sig = 4  # Hyperparameter

train_x = np.concatenate([np.linspace(1,20,n_samples).reshape(n_samples, 1), np.ones((n_samples,1))], axis=-1)
train_y = np.random.normal(w @ train_x.transpose(), sig)

# Plot the data points
plt.figure()
plt.scatter(train_x[:,0], train_y, label='Samples', s=3)
plt.title('Simulated Data')
plt.show()

w_est = train_y @ train_x @ np.linalg.inv(train_x.transpose() @ train_x)
# w_post_est = train_y @ train_x @ np.linalg.inv(train_x.transpose() @ train_x + 2*np.eye(2))

# Plot the line and data
plt.figure()
plt.scatter(train_x[:,0], train_y, label='Samples', s=3)
plt.title('Linear Regression')
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = w[:,1] + w[:,0] * x_vals
plt.plot(x_vals, y_vals, 'r-', label='Regression')
plt.legend()
plt.xlabel('w_true = %s, w_est = %s' % (w, w_est))
plt.show()

###
# Here using Tensorflow for linear regression by optimizing MLE
###

learning_r = 4e-2 # Try different learning rate, and find out what will happen
epochs = 1000

# Here defines Tensorflow Graph
x = tf.placeholder(tf.float64, shape=(None, 2))  # Here defines nodes for accepting input data
y = tf.placeholder(tf.float64, shape=(1, None))

w_est = tf.Variable(np.array([1,6], dtype="float64").reshape(1,2), name="weight") # Try different initial value

pred = tf.matmul(w_est, tf.transpose(x))

# Here defines the loss
loss = tf.reduce_sum(tf.square(y - pred))/(2*sig*train_x.shape[0])

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
    print('Estimated weight: %s' % (sess.run(w_est)))
    print('True weight: %s' % (w))
    plt.figure()
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.xlabel('Loss')
    plt.show()
