import cv2
from constants import RESIZE_IMAGE_LENGTH
import os

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def fetch_name(img_path):
    filepath, ext = os.path.splitext(img_path)
    if ext not in ['.jpg', '.png']:
        return None
    return filepath.split('/')[-1]

WHITE = [255, 255, 255]

def resize_and_grayscale(image_path):
    image_frame = cv2.imread(image_path)
    
    h = image_frame.shape[0]
    w = image_frame.shape[1]
    minh = min(RESIZE_IMAGE_LENGTH, h) 
    minw = min(RESIZE_IMAGE_LENGTH, w)

    if h > RESIZE_IMAGE_LENGTH or w > RESIZE_IMAGE_LENGTH:
        image_frame = cv2.resize(image_frame, (minh, minw))
        
    h = image_frame.shape[0]
    w = image_frame.shape[1]
    minh = min(RESIZE_IMAGE_LENGTH, h) 
    minw = min(RESIZE_IMAGE_LENGTH, w)
    bottom_border_width = RESIZE_IMAGE_LENGTH - minh
    right_border_width = RESIZE_IMAGE_LENGTH - minw
    image_frame= cv2.copyMakeBorder(image_frame, 0, bottom_border_width, 0, right_border_width, cv2.BORDER_CONSTANT,value=WHITE)
    image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    return image_frame