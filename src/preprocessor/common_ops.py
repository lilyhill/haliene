import cv2
import numpy as np
from Image import Image
from PreProcessor import PreProcessor
from ProcessState import ProcessState
from constants import HUE_PARAMS
from utils import show
import numpy.typing as npt


def hue_separate(preprocessor_runtime: PreProcessor, image: Image) -> npt.ArrayLike:
    hsv = cv2.cvtColor(image.frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([HUE_PARAMS['ilowH'], HUE_PARAMS['ilowS'], HUE_PARAMS['ilowV']])
    higher_hsv = np.array([HUE_PARAMS['ihighH'], HUE_PARAMS['ihighS'], HUE_PARAMS['ihighV']])

    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    # Apply the mask on the image to extract the original color
    frame = cv2.bitwise_and(image.frame, image.frame, mask=mask)
    
    if preprocessor_runtime.should_debug(image.name):
        show('hue_separate ' + image.name, frame)
    
    return Image(
        original_image_path=image.original_image_path,
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        name = image.name,
        original_frame = image.original_frame,
        state = ProcessState.HUE_SEPARATED,
    )


def get_full_img_binary(preprocessor_runtime: PreProcessor, image: Image):
    thresh = cv2.threshold(image, 20, 255,
	cv2.THRESH_BINARY)[1]

    if preprocessor_runtime.should_debug(image.name):
        show('thresholded on 20 ' + image.name, thresh)

    mask = cv2.bitwise_not(thresh)
    input_gray = cv2.cvtColor(image.original_image.frame, cv2.COLOR_BGR2GRAY)
    only_dust_img = cv2.bitwise_and(input_gray, mask, mask)
    only_dust_img_binary = cv2.threshold(only_dust_img, 100, 255,
	cv2.THRESH_BINARY)[1]
    return only_dust_img_binary


    kernel = np.ones((kernelsize, kernelsize),np.uint8)
    return cv2.dilate(eroded,kernel,iterations = iterations)


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


def cca(thresh):
    output = cv2.connectedComponentsWithStats(
	thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    output_img = INPUT_FRAME.copy()
    bigCount = 0
    print("centroids", centroids, len(centroids))
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format( i + 1, numLabels)
        # print a status message update for the current connected
        # component
        # print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # print("area", area)
        if area < 500 or area > 30000:
            continue
        (cX, cY) = centroids[i]
        
        x -= int(MARGIN/2)
        y -= int(MARGIN/2)
        w += MARGIN
        h += MARGIN
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.imshow("sperm thing", output_img)
        cv2.waitKey(0)
        # cv2.circle(output_img, (int(cX), int(cY)), 4, (255, 255, 255), -1)
        
        ## save the crop
        # make a UUID based on the host address and current time
        uuidOne = uuid.uuid1()
        crop = output_img[y:y+h,x:x+w]
        # cv2.imwrite('../images/output/ml_data/unlabelled/'+str(uuidOne)+'.jpg',crop)
        
        # componentMask = (labels == i).astype("uint8") * 255
        # cv2.imshow("Connected Component", componentMask)
        # cv2.waitKey(0)
        bigCount += 1
    print("bigCount", bigCount)
    return output_img