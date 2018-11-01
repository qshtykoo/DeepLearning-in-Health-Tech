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
from sklearn.metrics import roc_curve, auc
from scipy import interp

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
x_train = x_train.reshape(60000, 784)					# Reshape from (60000, 28, 28) to (60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')						# Use float32 for Keras
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(Y_train, 10)	# Turn label into one-hot vector
y_test = tf.keras.utils.to_categorical(Y_test, 10)

input_dim = x_train.shape[-1]							# Define the input dimension

# Start to model MLP
input = tf.keras.Input(shape=(input_dim, ))					# Input Layer
l1 = tf.keras.layers.Dense(512, activation='relu')(input)		# First Dense Layer
l2 = tf.keras.layers.Dropout(0.2)(l1)							# Droupout Layer
l3 = tf.keras.layers.Dense(10, activation='relu')(l2)			# Second Dense Layer
output = tf.keras.layers.Dense(10, activation='softmax')(l3)	# Softmax output

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
def confusion_matrix(y_pred, y_true):
    # Inputs: ground truth label and predicted label
    # Output: confusion matrix
    confusion_m = np.zeros((10,10)) #intialize confusion matrix for 10 classes, size of 10-by-10
    for i in range(len(y_true)):
        confusion_m[y_true[i]][y_pred[i]] = confusion_m[y_true[i]][y_pred[i]] + 1
    return confusion_m

# Calculate confusion matrix
y_true = Y_test     # convert one-hot label back to scalar
y_pred = np.argmax(model.predict(x_test), axis=-1)

cm = confusion_matrix(y_pred, y_true)	# get confusion matrix

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
# Plot PR curve of some of the classes, for example 0,1,2 in the example figure
n_classes = 3
for i, color, linestyle in zip(range(n_classes), colors, linestyles):
    l, = plt.plot(rec[i], pre[i], color=color, linestyle=linestyle, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, ave[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Classes 0, 1, 2')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.show()

#Here we plot ROC Curve
# Compute ROC curve and ROC area for each class
y_test = tf.keras.utils.to_categorical(y_true, 10)
y_score = tf.keras.utils.to_categorical(y_pred, 10)

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 10
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot of a ROC curve for a specific class
# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue', 'burlywood', 'darkorange', 'blue', 'red', 'black', 'green', 'orange']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()