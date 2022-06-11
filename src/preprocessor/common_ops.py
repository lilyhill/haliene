import cv2
import numpy as np
from Image import Image
from PreProcessor import PreProcessor
from ProcessState import ProcessState
from constants import HUE_PARAMS
from utils import show
from constants import MARGIN, MAX_CCA_AREA, MIN_CCA_AREA
import numpy.typing as npt


def hue_separate(preprocessor_runtime: PreProcessor, image: Image) -> npt.ArrayLike:
    hsv = cv2.cvtColor(image.frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([HUE_PARAMS['ilowH'], HUE_PARAMS['ilowS'], HUE_PARAMS['ilowV']])
    higher_hsv = np.array([HUE_PARAMS['ihighH'], HUE_PARAMS['ihighS'], HUE_PARAMS['ihighV']])

    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    # Apply the mask on the image to extract the original color
    frame = cv2.cvtColor(cv2.bitwise_and(image.frame, image.frame, mask=mask), cv2.COLOR_BGR2GRAY)
    if preprocessor_runtime.should_debug(image.name):
        show('hue_separate ' + image.name, frame)
    
    return Image(
        frame=frame,
        state = ProcessState.HUE_SEPARATED,
        original_image=image.original_image
    )


def get_dustful(preprocessor_runtime: PreProcessor, image: Image):
    hue_highlighted = cv2.threshold(image.frame, 20, 255,
	cv2.THRESH_BINARY)[1]

    if preprocessor_runtime.should_debug(image.name):
        show('thresholded on 20 ' + image.name, hue_highlighted)

    remove_hue_highlighted_mask = cv2.bitwise_not(hue_highlighted)
    input_gray_naturally_inverted = cv2.cvtColor(image.original_image.frame, cv2.COLOR_BGR2GRAY)
    input_naturally_inverted_binary = cv2.threshold(input_gray_naturally_inverted, 100, 255,
	cv2.THRESH_BINARY)[1]
    input_binary = cv2.bitwise_not(input_naturally_inverted_binary)

    if preprocessor_runtime.should_debug(image.name):
        show('input_binary ' + image.name, input_binary)
    
    ## Full image - big blobs formed from hue seperation = dusty image
    dustful = cv2.bitwise_and(input_binary, remove_hue_highlighted_mask, mask = remove_hue_highlighted_mask)  

    if preprocessor_runtime.should_debug(image.name):
        show('dustful' + image.name, dustful)
    
    return Image(
        frame=dustful,
        state = ProcessState.DUSTFUL,
        original_image=image.original_image
    ), Image(
        frame=hue_highlighted,
        state = ProcessState.HUE_HIGHLIGHTED,
        original_image=image.original_image
    ), Image(
        frame=input_binary,
        state = ProcessState.THRESOLDED,
        original_image=image.original_image
    )

def inpainting(input_img_path, mask_img_path, radius = 3, method = None):
    flags = cv2.INPAINT_TELEA
    if method == "ns":
        flags = cv2.INPAINT_NS
    image = cv2.imread(input_img_path)
    mask_img = cv2.imread(mask_img_path)
    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    print("mask", mask.shape, type(mask))
    output = cv2.inpaint(image, mask, radius, flags=flags)
    return output


def cca(preprocessor_runtime: PreProcessor, binary_image: Image):
    (numLabels, _, stats, centroids) = cv2.connectedComponentsWithStats(
	binary_image.frame, 4, cv2.CV_32S)
    output_img_frame = binary_image.original_image.frame.copy()
    bigCount = 0
    print("centroids", centroids, len(centroids))
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        else:
            # otherwise, we are examining an actual connected component
            text = "examining component {}/{}".format( i + 1, numLabels)
        
        print("[INFO] {}".format(text))
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_CCA_AREA or area > MAX_CCA_AREA:
            continue
        # (cX, cY) = centroids[i]
        
        x -= int(MARGIN/2)
        y -= int(MARGIN/2)
        w += MARGIN
        h += MARGIN
        cv2.rectangle(output_img_frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # cv2.circle(output_img, (int(cX), int(cY)), 4, (255, 255, 255), -1)
        
        ## save the crop
        cropped_image_name = preprocessor_runtime.get_mldata_unlabelled_dir() + '/' + binary_image.original_image.name + '_' + str(bigCount) + '.jpg'
        print('cropped_image_name', cropped_image_name)
        crop = binary_image.original_image.frame[y:y+h,x:x+w]
        cv2.imwrite(cropped_image_name, crop)
        bigCount += 1
    print("bigCount", bigCount)
    return Image(
        frame=output_img_frame,
        state = ProcessState.BOUNDING_BOXED,
        original_image = binary_image.original_image
    )
