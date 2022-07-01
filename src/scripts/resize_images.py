import os
import cv2
RESIZE_IMAGE_LENGTH = 120

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
    # image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    return image_frame


## run this from inside labelled_sorted
def main():
    here_path = os.getcwd()
    print("here", here_path)
    for dir in os.listdir(here_path):
        image_dir = os.path.join(here_path, dir)
        print("here", dir, image_dir)
        if not os.path.isdir(image_dir):
            continue
        print("dir", dir, image_dir)
        for image_file in os.listdir(image_dir):
            if image_file == '.DS_Store':
                continue
            image_file_path = os.path.join(image_dir, image_file)
            if not os.path.isfile(image_file_path):
                continue
            print("image_file", image_file, image_file_path)
            image_frame = resize_and_grayscale(image_file_path)
            cv2.imwrite(image_file_path, image_frame) 


main()