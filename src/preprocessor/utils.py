import cv2
import os

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def fetch_name(img_path):
    filepath, ext = os.path.splitext(img_path)
    if ext not in ['.jpg', '.png']:
        return None
    return filepath.split('/')[-1]
