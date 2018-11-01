import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
import matplotlib.gridspec as gridspec

# This is just a template to inspire you how to visualize the filter and output of convolutional layers in Keras.
# You need to adapt the code for your model specifically. 

# Here is your code for CIFAR10 classification using CNN
...

# After the training, you save the model in case you want to show your results without training it again.
model.save('my_model.h5')

# Now load the model
model = keras.models.load_model('my_model.h5')
model.summary()  # Show details of your model

# Let us first visualize one output of a convolutional layer
vis_out_model = keras.Model(inputs = model.input, outputs=model.layers[???].output) # You need to find out which conv layer you want to visualize and change that ???

vis_act_model = keras.Model(inputs = model.input, outputs=model.layers[???].output) # You need to find out which activation layer you want to visualize and change that ???

vis_out = vis_out_model.predict(x_test)[???]   # ??? which data you want to show
vis_act = vis_act_model.predict(x_test)[???]

# Now the expected shape of vis_out and vis_act should be some (w, h, d), that you have d numbered w*h sized images to show
# Show using subplot

fig, axs = plt.subplots(nrows=???, ncols=???,
                        gridspec_kw={'top':1.0, 'bottom':0.0, 'left':0.0,'right':1.0,'wspace':0.02, 'hspace':0.02},
                        squeeze=True, figsize=(16,2))
for i in range(???):
    for j in range(???):
        axs[i][j].imshow(vis_out[:,:,i*??? + j])
        axs[i][j].axis('off')
        #axs[i][j].invert_yaxis()
        axs[i][j].set_yticklabels([])
        axs[i][j].set_xticklabels([])
plt.savefig('vis_out.eps')
plt.show()

# The same code for showing vis_act

# Now we visualize the weights of filters

# Fetch the weight from a layer
weight = model.layers[???].get_weights()   # replace ?? with the layer you want to show

# Plot the images using subplot

