import numpy as np
from matplotlib import pyplot as plt
import cv2
import noise


img = cv2.imread('img/test1.bmp')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
labels = gray_image / 127
img2 = noise.gauss(gray_image,var=1000)

plt.subplot(121), plt.imshow(gray_image, cmap='gray'), plt.title('GRAY')
plt.subplot(122), plt.imshow(img2, cmap='gray'), plt.title('NOISE')

plt.show()
