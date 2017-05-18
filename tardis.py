import sys
import numpy as np
from skimage import color
from scipy import misc
from scipy import ndimage
from scipy.ndimage import measurements,morphology
import matplotlib.pyplot as plt

minU = 0.1
maxU = 0.5
minV = -0.5
maxV = 0.0

def displayBinaryImage(image):
    plt.imshow(1-image, cmap = 'gray')

def smoothBinaryImage(image):
    kernel = ndimage.generate_binary_structure(2,1).astype(np.int)
    iterations = 15

    smoothed = morphology.binary_closing(binary, kernel ,iterations=iterations)
    return smoothed

# Returns (x1, y1, x2, y2)
def findLargestObject(binaryImage):
    labels, nbr_objects = measurements.label(binaryImage)
    print "Number of objects:", nbr_objects
    objects = measurements.find_objects(labels)

    maxArea = 0
    largestObject = ()

    for o in objects:
        x1 = o[1].start
        x2 = o[1].stop
        y1 = o[0].start
        y2 = o[0].stop
        area = (x2-x1+1)*(y2-y1+1)

        if area > maxArea:
            maxArea = area
            largestObject = o

    return (largestObject[1].start, largestObject[0].start, largestObject[1].stop, largestObject[0].stop)

img = misc.imread("tardis.jpg")
hsv = color.rgb2hsv(img)
yuv = color.rgb2yuv(img)

binary = np.ones((img.shape[0], img.shape[1]), dtype='uint8')

binary[yuv[:,:,1] < minU] = 0
binary[yuv[:,:,1] > maxU] = 0

binary[yuv[:,:,2] < minV] = 0
binary[yuv[:,:,2] > maxV] = 0

smoothed = smoothBinaryImage(binary)

object = findLargestObject(smoothed)

plt.figure(figsize=(10,8))

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

(x1, y1, x2, y2) = object
plt.plot((x1, x2), (y1, y1), color='red')
plt.plot((x1, x2), (y2, y2), color='red')
plt.plot((x1, x1), (y1, y2), color='red')
plt.plot((x2, x2), (y1, y2), color='red')

plt.show()
