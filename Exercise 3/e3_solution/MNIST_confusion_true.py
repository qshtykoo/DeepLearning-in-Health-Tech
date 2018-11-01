'''
ELEC- E8739 AI in Health Technologies. Exercise III
Confusion matrix and ROC, PR curve plotting
Code by Zheng Zhao
Aalto University
zz@zabemon.com
Date: Sep 3, 2018
'''
# Import necessary dependencies, if you dont have some of them, use pip install or conda install
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plot_lib as plot_lib
from sklearn.metrics import confusion_matrix

mnist = tf.keras.datasets.mnist # If you have not MNIST installed, it will download automatically. 

(X_train, Y_train),(X_test, Y_test) = mnist.load_data() # Load

plt.figure(figsize=(8,8))                               # Plot some of the dataset
plt.suptitle('Some samples of MNIST dataset')
for i in range(16):
    idx = np.random.randint(60000)
    plt.subplot(4,4,i+1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title("Class %s" % (Y_train[idx]))
plt.show()

x_train, x_test = X_train / 255.0, X_test / 255.0       # Normalize them to [0,1]
x_train = x_train.reshape(60000, 784)         # Reshape from (60000, 28, 28) to (60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')           # Use float32 for Keras
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(Y_train, 10)  # Turn label into one-hot vector
y_test = tf.keras.utils.to_categorical(Y_test, 10)

input_dim = x_train.shape[-1]             # Define the input dimension

# Start to model MLP
input = tf.keras.Input(shape=(input_dim, ))         # Input Layer
l1 = tf.keras.layers.Dense(512, activation='relu')(input)   # First Dense Layer
l2 = tf.keras.layers.Dropout(0.2)(l1)             # Droupout Layer
l3 = tf.keras.layers.Dense(10, activation='relu')(l2)     # Second Dense Layer
output = tf.keras.layers.Dense(10, activation='softmax')(l3)  # Softmax output

model = tf.keras.Model(input, output)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# Start to training
model.fit(x_train, y_train, 
  batch_size=128,
  epochs=1,
  validation_data=(x_test, y_test))   # only 1 iteration just for

# Save the trained model
model.save('e1-mnist-model.h5')

# Here you need to define a function that calculate confusion matrix
# def confusion_matrix(y_pred, y_true):
#     # Inputs: ground truth label and predicted label
#     # Output: confusion matrix
#     pass

# Calculate confusion matrix
y_true = np.argmax(y_test, axis=-1)     # convert one-hot label back to scalar
y_pred = np.argmax(model.predict(x_test), axis=-1)

cm = confusion_matrix(y_pred, y_true) # Yield confusion matrix

# Plot confusion matrix
plot_lib.plot_confusion_matrix(cm,
                               normalize=True,
                               classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Here we plot the PR curve
# F1 iso
plt.figure()
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []

for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')

pre, rec, ave = plot_lib.get_mean_prc(tf.keras.utils.to_categorical(y_true, 10), tf.keras.utils.to_categorical(y_pred, 10))
colors = ['burlywood', 'darkorange', 'blue']
linestyles = ['-', '--', ':']
for i in range(3):  # just choose three of the classes to show
    l, = plt.plot(rec[i], pre[i], color=colors[i], lw=2, linestyle=linestyles[i])
    lines.append(l)
    labels.append('Classes %d (area = %0.2f)' % (i, ave[i]))

fig = plt.gcf()
#fig.subplots_adjust(top=1.0, bottom=0.0)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(lines, labels)
#plt.legend(lines, labels, prop=dict(size=14))
plt.tight_layout()
plt.show()