
from traceback import FrameSummary
import cv2
import numpy as np

INPUT_IMG = '../images/input/sample1.jpg'
INPUT_FRAME = cv2.imread(INPUT_IMG)
HUE_PARAMS = {
    'ilowH' : 65,
    'ihighH' : 156,
    'ilowS' : 80,
    'ihighS' : 255,
    'ilowV' : 0,
    'ihighV' : 189,
}

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def inpainting(input_img_path, mask_img_path, radius = 3, method = None):
    flags = cv2.INPAINT_TELEA
    if method == "ns":
        flags = cv2.INPAINT_NS
    image = cv2.imread(input_img_path)
    mask_img = cv2.imread(mask_img_path)
    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    print("mask", mask.shape, type(mask))
    output = cv2.inpaint(image, mask, radius, flags=flags)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)

def hue_separate():
    
    hsv = cv2.cvtColor(INPUT_FRAME, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([HUE_PARAMS['ilowH'], HUE_PARAMS['ilowS'], HUE_PARAMS['ilowV']])
    higher_hsv = np.array([HUE_PARAMS['ihighH'], HUE_PARAMS['ihighS'], HUE_PARAMS['ihighV']])

    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    # Apply the mask on the image to extract the original color
    frame = cv2.bitwise_and(INPUT_FRAME, INPUT_FRAME, mask=mask)
    
    # cv2.imshow('image', frame)
    cv2.imwrite('../images/output/asds.jpg', frame)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def get_full_img_binary(gray):
    thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY)[1]
    show("thresh", thresh)
    cv2.imwrite("../images/output/asds_thresh.jpg", thresh)
    mask = cv2.bitwise_not(thresh)
    input_gray = cv2.cvtColor(INPUT_FRAME, cv2.COLOR_BGR2GRAY)
    only_dust_img = cv2.bitwise_and(input_gray, mask, mask)
    only_dust_img_binary = cv2.threshold(only_dust_img, 100, 255,
	cv2.THRESH_BINARY)[1]
    return only_dust_img_binary

def erode(dustful_img, kernelsize = 8, iterations = 2):
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    return cv2.erode(dustful_img, kernel, iterations = iterations)

def dilate(eroded, kernelsize = 3, iterations = 10):
    kernel = np.ones((kernelsize, kernelsize),np.uint8)
    return cv2.dilate(eroded,kernel,iterations = iterations)



def main():
    gray = hue_separate()
    cv2.imwrite('../images/output/asds_gray.jpg', gray)
    full_img_binary_img = get_full_img_binary(gray)
    # full_img_binary_img = cv2.cvtColor(full_img_binary_img, cv2.COLOR_BGR2GRAY)
    full_img_binary = '../images/output/full_img_binary.jpg'
    cv2.imwrite(full_img_binary, full_img_binary_img)
    asds_mask = cv2.imread('../images/output/asds_thresh.jpg')
    asds_mask = cv2.cvtColor(asds_mask, cv2.COLOR_BGR2GRAY)
    show("asds_mask", asds_mask)
    dustful = cv2.bitwise_or(full_img_binary_img, asds_mask, asds_mask)
    show("dustful", dustful)
    dustful = cv2.bitwise_not(dustful)
    cv2.imwrite("../images/output/bdsdds.jpg", dustful)


    # infill = inpainting(bssdds, asds_mask)

    ## morphological operations on bdds
    eroded = erode(dustful)
    show("erosion", eroded)
    cv2.imwrite("../images/output/bdds_eroded.jpg", eroded)

    bdds_dilated = dilate(eroded)
    show("dilation", bdds_dilated)
    cv2.imwrite("../images/output/bdds.jpg", bdds_dilated)


    ## remove noise from asds_mask
    asds_mask = cv2.imread('../images/output/asds_thresh.jpg')
    show("asds_mask", asds_mask)
    eroded = erode(asds_mask, kernelsize=6, iterations=3)
    show("erosion1", eroded)
    cv2.imwrite("../images/output/asds_mask_eroded.jpg", eroded)

    dilated = dilate(eroded, kernelsize=6, iterations=8)
    show("dilation1", dilated)
    cv2.imwrite("../images/output/asds_mask_dilated.jpg", dilated)

    asds_mask_clean = cv2.bitwise_and(asds_mask, dilated, asds_mask)
    show("asds_mask_clean", asds_mask_clean)
    cv2.imwrite("../images/output/asds_mask_clean.jpg", asds_mask_clean)

    ## club asds_mask_clean and bdds and form bddsas mask 
    print("asds", asds_mask_clean.shape, bdds_dilated.shape)
    asds_mask_clean_gray = cv2.cvtColor(asds_mask_clean, cv2.COLOR_BGR2GRAY)
    bddsas_mask = cv2.bitwise_or(asds_mask_clean_gray, bdds_dilated, bdds_dilated)
    show("bddsas_mask", bddsas_mask)
    cv2.imwrite("../images/output/bddsas_mask.jpg", bddsas_mask)


    ## use bddsas mask to create final dirtless image.




    

    


if __name__ == '__main__':
    main()