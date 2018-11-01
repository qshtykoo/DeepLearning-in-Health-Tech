import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
import math

# Load some fancy pictures and convert into gray scale
plt.figure()
cat = io.imread('haha.jpg')
plt.imshow(cat, cmap='gray')


def flip_up_down(matrix):
    cols = matrix.shape[1]
    for i in range(cols):
        matrix[:, i,: ] = matrix[::-1, i,:]
    return matrix

def flip_left_right(matrix):
    rows = matrix.shape[0]
    for i in range(rows):
        matrix[i, :,:] = matrix[i, ::-1,:]
    return matrix

cat_new = flip_left_right(cat)
plt.figure()
plt.imshow(cat_new, cmap='gray')
plt.show()
