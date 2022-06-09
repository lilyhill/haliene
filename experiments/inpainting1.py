

import cv2
import numpy as np
 
def inpainting(input_img_path, radius = 3, method = None):
    flags = cv2.INPAINT_TELEA
    # check to see if we should be using the Navier-Stokes (i.e., Bertalmio
    # et al.) method for inpainting
    if method == "ns":
        flags = cv2.INPAINT_NS
    image = cv2.imread(input_img_path)
    mask_image = cv2.imread('images/output/only_living.jpg')
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("mask", mask.shape, type(mask))
    output = cv2.inpaint(image, mask, radius, flags=flags)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.imshow("Output", output)
    cv2.waitKey(0)

if __name__ == '__main__':
    inpainting(input_img_path='images/input/only_living.jpg', radius=20)    
