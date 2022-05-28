import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('images/input/only_living_inpainted.jpg')
blur = cv.blur(img,(10,10))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
cv.imwrite('images/output/only_living_inpainted_blur.jpg', blur)