import cv2
import matplotlib.pyplot as plt
import numpy as np

image1 = cv2.imread("val_set/1/1a.png", flags=cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("val_set/1/1b.png", flags=cv2.IMREAD_GRAYSCALE)
delta = image1-image2
plt.imshow(image1, cmap='gray')
plt.show()
plt.imshow(image2, cmap='gray')
plt.show()
plt.imshow(delta, cmap='gray')
plt.show()