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
n_samples = 1000 # Number of samples (datapoints)

w = np.array([2,5]).reshape(1,2)	 # Ground truth weight
sig = 4  # Hyperparameter (in the exercise, it is sigma = 1)

train_x = np.concatenate([np.linspace(1,20,n_samples).reshape(n_samples, 1), np.ones((n_samples,1))], axis=-1)
train_y = np.dot(train_x,w.T) + np.random.normal(0, sig, size=(n_samples,1))  #y = xW^T + epsilon

# Plot the data points
plt.figure()
plt.scatter(train_x[:,0], train_y, label='Samples', s=3)
plt.title('Simulated Data')
plt.show()

xo = train_x
xt = xo.T
xtx = np.dot(xt,xo)-1  #the order of matrix is important
xty = np.dot(xt,train_y)

w_est = np.linalg.solve(xtx,xty) # Here is the estimated weight results from 1.1

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
epochs = 3000   # Iteration

# Here defines Tensorflow Graph
x = tf.placeholder("float", shape=(None,2))  # Here defines nodes for accepting input data
y = tf.placeholder("float", shape=(None,1))

w_est = tf.Variable(tf.ones([2,1]), name="weight") # Here defines estimated weight to be optimized. Try different initial value

pred = tf.matmul(x,w_est)  # Here defines predicted y

# Here defines the mean square loss
# tf.reduce_sum() Computes the sum of elements across dimensions of a tensor
loss = tf.reduce_sum(tf.square(y-pred)/(2*sig*train_x.shape[0])) #for first-order derivative optimization it is needed to be normalized for SGD method
#loss = tf.reduce_sum(tf.square(y-pred))
#loss = tf.log(loss) #why only log-loss work

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
        #sess.run(w_est, feed_dict={x:train_x, y:train_y})
        #sess.run(pred, feed_dict={x:train_x, y:train_y})
        #sess.run(loss, feed_dict={x:train_x, y:train_y})
        sess.run(optimizer, feed_dict={x: train_x, y: train_y})
        history[e] = sess.run(loss, feed_dict={x: train_x, y: train_y})
        print("loss: %f, w_est: %s" % (history[e], sess.run(w_est)))

    # Print and plot results
    print('Estimated weight: %s' % (sess.run(w_est)))
    print('True weight: %s' % (w))
    plt.figure()
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
