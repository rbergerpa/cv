import sys
import numpy as np
from scipy import misc
from skimage import color
import matplotlib.pyplot as plt

minHue = .5
maxHue = .6
minSaturation = .6
maxSaturation = .7

img = misc.imread("tardis.jpg")
hsv = color.rgb2hsv(img)
binary = np.zeros_like(img)

binary[hsv[:,:,0] < minHue] = 255
binary[hsv[:,:,0] > maxHue] = 255

binary[hsv[:,:,1] < minSaturation] = 255
binary[hsv[:,:,1] > maxSaturation] = 255

plt.subplot(2,2,1)
plt.imshow(img)

plt.subplot(2,2,2)
plt.imshow(binary)

plt.show()

