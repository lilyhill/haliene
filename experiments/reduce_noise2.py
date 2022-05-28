#!/bin/python3.9

import cv2
import numpy as np

src = cv2.imread('sample1.jpg', cv2.IMREAD_GRAYSCALE)

# convert to binary by thresholding
ret, binary_map = cv2.threshold(src,127,255,0)

# do connected components processing
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN] 
areas = stats[1:,cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 100:   #keep
        result[labels == i + 1] = 255

cv2.imshow("Binary", binary_map)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("sample1_rn2.jpg", result)