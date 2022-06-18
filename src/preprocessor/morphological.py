import numpy as np
import cv2
from Image import Image
from ProcessState import ProcessState
from utils import show


def erode(preprocessor_runtime, image: Image, kernelsize = 8, iterations = 2):
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    frame = cv2.erode(image.frame, kernel, iterations = iterations)
    if preprocessor_runtime.should_debug(image.name):
        show('erode ' + image.name+ ' ' + image.state.value, frame)
    return Image(
        frame=frame,
        state = ProcessState.ERODED,
        original_image=image.original_image
    )


def dilate(preprocessor_runtime, image, kernelsize = 3, iterations = 10):
    kernel = np.ones((kernelsize, kernelsize),np.uint8)
    frame = cv2.dilate(image.frame, kernel, iterations = iterations)
    if preprocessor_runtime.should_debug(image.name):
        show('dilate ' + image.name + ' ' + image.state.value, frame)
    return Image(
        frame=frame,
        state = ProcessState.DILATED,
        original_image=image.original_image
    )


def reduce_noise_from_dustful_image(preprocessor_runtime, dustful_image: Image):
    eroded_dustful_image = erode(preprocessor_runtime=preprocessor_runtime, image=dustful_image)
    dilated_dustful_image = dilate(preprocessor_runtime=preprocessor_runtime, image=eroded_dustful_image)
    only_bigdusty_dustful_frame = cv2.bitwise_and(dustful_image.frame, dilated_dustful_image.frame, mask=dilated_dustful_image.frame)
    if preprocessor_runtime.should_debug(dustful_image.name):
        show('morphed ' + dustful_image.name + ' ' + dustful_image.state.value, only_bigdusty_dustful_frame)
    return Image(
        frame=only_bigdusty_dustful_frame,
        state = ProcessState.MORPHED,
        original_image=dustful_image.original_image
    )


def reduce_noise_from_hue_highlighted_image(preprocessor_runtime, hue_highlighted: Image):
    eroded_hue_highlighted = erode(preprocessor_runtime=preprocessor_runtime, image=hue_highlighted, kernelsize=6, iterations=0)
    dilated_hue_highlighted = dilate(preprocessor_runtime=preprocessor_runtime, image=eroded_hue_highlighted, kernelsize=6, iterations=8)
    dustless_hue_highlighted_frame = cv2.bitwise_and(hue_highlighted.frame, dilated_hue_highlighted.frame, mask=dilated_hue_highlighted.frame)
    if preprocessor_runtime.should_debug(hue_highlighted.name):
        show('morphed ' + hue_highlighted.name + ' ' + hue_highlighted.state.value, dustless_hue_highlighted_frame)
    return Image(
        frame=dustless_hue_highlighted_frame,
        state = ProcessState.MORPHED,
        original_image=hue_highlighted.original_image
    )
