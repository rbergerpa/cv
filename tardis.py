import sys
import numpy as np
from scipy import misc
from skimage import color
import matplotlib.pyplot as plt

minHue = .5
maxHue = .6
minSaturation = .6
maxSaturation = .7

def displayBinaryImage(image):
    plt.imshow(1-image, cmap = 'gray')

img = misc.imread("tardis.jpg")
hsv = color.rgb2hsv(img)

binary = np.ones((img.shape[0], img.shape[1]), dtype='uint8')

binary[hsv[:,:,0] < minHue] = 0
binary[hsv[:,:,0] > maxHue] = 0

binary[hsv[:,:,1] < minSaturation] = 0
binary[hsv[:,:,1] > maxSaturation] = 0

plt.subplot(2,2,1)
plt.imshow(img)

plt.subplot(2,2,2)
displayBinaryImage(binary)

plt.show()

