import cv2
import numpy as np
from Image import Image
from ProcessState import ProcessState
from constants import HUE_PARAMS
from utils import show
from constants import MARGIN_DICT, MAX_CCA_AREA, MIN_CCA_AREA, SIZE_AREA_DICT
import numpy.typing as npt
import os
import time


def hue_separate(preprocessor_runtime, image: Image) -> npt.ArrayLike:
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


def get_dustful(preprocessor_runtime, image: Image):
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


def cca(preprocessor_runtime, binary_image: Image):
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
	binary_image.frame, 4, cv2.CV_32S)
    output_img_frame = binary_image.original_image.frame.copy()
    bigCount = 0
    print("centroids", centroids, len(centroids))
    crop_images_dir = preprocessor_runtime.get_mldata_unlabelled_dir() + '/' + binary_image.original_image.name
    print("crop_images_dir", crop_images_dir)
    try:
        os.mkdir(crop_images_dir)
        os.mkdir(crop_images_dir+'/big')
        os.mkdir(crop_images_dir+'/medium')
        os.mkdir(crop_images_dir+'/small')
    except FileExistsError as fee:
        print("fee", str(fee))
        pass
    except Exception as e:
        raise Exception("error while creating folder in cca " + str(e))
    for i in range(0, numLabels):
        output_img_frame_for_show = binary_image.original_image.frame.copy()
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
        if area < MARGIN_DICT['area_threshold_min']:
            margin = MARGIN_DICT['small_area_margin']
        elif area > MARGIN_DICT['area_threshold_max']:
            margin = MARGIN_DICT['large_area_margin']
        else:
            margin = 0
        x -= int(margin/2)
        y -= int(margin/2)
        w += margin
        h += margin
        cv2.rectangle(output_img_frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # cv2.rectangle(output_img_frame_for_show, (x, y), (x + w, y + h), (255, 255, 255), 3)
        print("area ", area, w, h)
        # show(f'everytime {i} ', output_img_frame_for_show)
        # cv2.circle(output_img, (int(cX), int(cY)), 4, (255, 255, 255), -1)
        # componentMask = (labels == i).astype("uint8") * 255
        # cv2.imshow("Connected Component", componentMask)
        # cv2.waitKey(0)
        
        ## save the crop
        if preprocessor_runtime.should_crop_image:
            # below we sub-classify based on size. so that one can easily look at cropped image and tell if it is big, small or medium
            middle_folder = 'big'
            if area < SIZE_AREA_DICT['thresh1']:
                middle_folder = 'small'
            elif area < SIZE_AREA_DICT['thresh2']:
                middle_folder = 'medium'
            cropped_image_name = crop_images_dir + '/' + middle_folder + '/' + binary_image.original_image.name + '_' + str(bigCount) + '.jpg'
            print('cropped_image_name', cropped_image_name)
            crop = binary_image.original_image.frame[y:y+h,x:x+w]
            # remove crops which are all zeros
            if np.all(crop == 0):
                continue
            failure_count = 0
            try:
                while not cv2.imwrite(cropped_image_name, crop) and failure_count < 5:
                    failure_count += 1
                    time.sleep(1)
                    print("sleeping", cropped_image_name)
                if failure_count == 5:
                    print("could not crop", cropped_image_name)
            except Exception as e:
                print("could not crop due to exception", cropped_image_name)
        
        bigCount += 1
    print("bigCount", bigCount)
    return Image(
        frame=output_img_frame,
        state = ProcessState.BOUNDING_BOXED,
        original_image = binary_image.original_image
    )
