from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from scipy import misc
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

mod = SourceModule("""
__global__ void segment(unsigned char* rgb, unsigned char* binary, int width, int height) {
    const float Wr = 0.299;
    const float Wg = 0.587;
    const float Wb = 0.114;
    const float Ub = 0.492;
    const float Vr = 0.877;
    const float S = 1/255.0;
    const float minU = 0.1;
    const float maxU = 0.5;
    const float minV = -0.5;
    const float maxV = 0.0;

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < width && iy < height) {
        unsigned int idx = iy*width + ix;

        unsigned char* pixel = rgb + 3*idx;
        float r = S * *pixel++;
        float g = S * *pixel++;
        float b = S * *pixel++;

        float y = Wr*r + Wg*g + Wb*b;
        float u = Ub*(b - y);
        float v = Vr*(r - y);

        unsigned char* out = binary  + idx;
        *out = u >= minU && u <= maxU && v >= minV && v <= maxV;
    }
}
""")

segment = mod.get_function("segment")

img = misc.imread("../tardis.jpg")

SIZE_Y = img.shape[0]
SIZE_X = img.shape[1]

BLOCK_X = 128
BLOCK_Y = 1

GRID_X = (SIZE_X + BLOCK_X -1)/BLOCK_X
GRID_Y = (SIZE_Y + BLOCK_Y -1)/BLOCK_Y

binary = np.zeros(shape=img.shape[0:2], dtype='int8')

start = time.time()

segment(drv.In(img), drv.Out(binary), np.int32(SIZE_X), np.int32(SIZE_Y), block=(BLOCK_X, BLOCK_Y, 1), grid=(GRID_X, GRID_Y, 1))

end = time.time()
print "time", end - start

def displayBinaryImage(image):
    plt.imshow(1-image, cmap = 'gray')

plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.axis('off')
plt.imshow(img)

plt.subplot(2,1,2)
plt.axis('off')
displayBinaryImage(binary)

plt.show(block=False)
