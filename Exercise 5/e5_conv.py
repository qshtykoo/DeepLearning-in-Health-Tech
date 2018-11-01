"""
ELEC- E8739 AI in Health Technologies. Exercise V
Template for simple convolutional operation for image processing
Aalto University
Oct 5, 2018
Image are from internet
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color

# Load some fancy pictures and convert into gray scale
plt.figure()
cat = color.rgb2gray(io.imread('haha.jpg'))
plt.imshow(cat, cmap='gray')


# Here you define the function for convolutional operation
def conv(img, filter, stride):
    """
    img: Input image
    filter: The filter for convolution
    stride: stride
    """
    # Calculate the expected size of output
    input_shape = img.shape
    filter_shape = filter.shape
    out_shape = np.subtract(input_shape, filter_shape)/stride + 1  # See equation in Section 2.6

    out_shape = out_shape.astype(int)
    out = np.zeros(shape=out_shape)

    # Start to do convolution
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            out[i,j] = np.sum(np.multiply(img[i:i+3, j:j+3], filter))

    return out

kernel_1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) # Try different kernel
kernel_2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
stride = 1

img_filtered_2 = conv(cat, kernel_2, stride)
img_filtered_1 = conv(cat, kernel_1, stride)

plt.figure()
plt.imshow(img_filtered_1, cmap='gray')    # Show the result


plt.figure()
plt.imshow(img_filtered_2, cmap="gray")

