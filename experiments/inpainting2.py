

from traceback import FrameSummary
import cv2
import numpy as np
 
def inpainting(input_img_path, radius = 5, method = None, limit = 5):
    flags = cv2.INPAINT_TELEA
    # check to see if we should be using the Navier-Stokes (i.e., Bertalmio
    # et al.) method for inpainting
    if method == "ns":
        flags = cv2.INPAINT_NS
    image = cv2.imread(input_img_path)
    mask_image = cv2.imread('images/output/global_threshold.png')
    mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    print("mask", mask.shape, type(mask))

    for i in range(limit):
        output = cv2.inpaint(image, mask, radius, flags=flags)
        output_img_path = f"images/output/only_living_inpainted{i}.jpg"
        cv2.imwrite(output_img_path, output)
        mask = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        image = output
        print("i done", i)

    cv2.imshow("Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Output", output)
    cv2.waitKey(0)

if __name__ == '__main__':
    inpainting(input_img_path='images/input/only_living.jpg')    
