'''
ELEC- E8739 AI in Health Technologies. Kickstart for exercise I
Code by Zheng Zhao
Date: Sep 3, 2018
'''
# Import necessary dependencies, if you dont have some of them, use pip install or conda install
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
	epochs=5, 
	validation_data=(x_test, y_test))

# Save the trained model
model.save('e1-mnist-model.h5')

# Evaluate the results on test set
acc = model.evaluate(x_test, y_test)
y_pred = np.argmax(model.predict(x_test), axis=-1)
print('Test score:', acc[0])
print('Test accuracy:', acc[1])

# Plot some random records in the test set, and show their predicted class and original class
plt.figure(figsize=(8,8))
plt.suptitle('Original and Predicted label')
for i in range(16):										# Plot some of the dataset
    idx = np.random.randint(10000)
    plt.subplot(4,4,i+1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title("Ori/Pred %s/%s" % (Y_test[idx], y_pred[idx]))
plt.show()
