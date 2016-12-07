import sys
import numpy as np
from scipy import misc
from skimage import color
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,morphology

minHue = .5
maxHue = .6
minSaturation = .6
maxSaturation = .7

def displayBinaryImage(image):
    plt.imshow(1-image, cmap = 'gray')

def smoothBinaryImage(image):
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    iterations = 15

    smoothed = morphology.binary_closing(binary, kernel ,iterations=iterations)
    return smoothed

img = misc.imread("tardis.jpg")
hsv = color.rgb2hsv(img)

binary = np.ones((img.shape[0], img.shape[1]), dtype='uint8')

binary[hsv[:,:,0] < minHue] = 0
binary[hsv[:,:,0] > maxHue] = 0

binary[hsv[:,:,1] < minSaturation] = 0
binary[hsv[:,:,1] > maxSaturation] = 0

smoothed = smoothBinaryImage(binary)

labels, nbr_objects = measurements.label(smoothed)
print "Number of objects:", nbr_objects
objects = measurements.find_objects(labels)

maxArea = 0
maxAreaX1 = 0
maxAreaX2 = 0
maxAreaY1 = 0
maxAreaY2 = 0

for o in objects:
    x1 = o[1].start
    x2 = o[1].stop
    y1 = o[0].start
    y2 = o[0].stop
    area = (x2-x1+1)*(y2-y1+1)

    if area > maxArea:
        maxArea = area
        maxAreaX1 = x1
        maxAreaX2 = x2
        maxAreaY1 = y1
        maxAreaY2 = y2

plt.subplot(2,2,1)
plt.axis('off')
plt.imshow(img)

plt.subplot(2,2,2)
plt.axis('off')
displayBinaryImage(binary)

plt.subplot(2,2,3)
plt.axis('off')
displayBinaryImage(smoothed)

plt.subplot(2,2,4)
plt.axis([0, img.shape[1], img.shape[0], 0])
plt.axis('off')
plt.imshow(img)
plt.plot([maxAreaX1, maxAreaX2], [maxAreaY1, maxAreaY1], color='red')
plt.plot([maxAreaX1, maxAreaX2], [maxAreaY2, maxAreaY2], color='red')
plt.plot([maxAreaX1, maxAreaX1], [maxAreaY1, maxAreaY2], color='red')
plt.plot([maxAreaX2, maxAreaX2], [maxAreaY1, maxAreaY2], color='red')
plt.show()
