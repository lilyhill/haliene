import cv2
import numpy as np

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def erode_living(input_img_path):
    img = cv2.imread(input_img_path)
    show("img", img)
    print("imgshape", img.shape)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show("gray", gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    show("thresh", thresh)
    kernel = np.ones((10, 10),np.uint8)
    erosion = cv2.erode(thresh, kernel,iterations = 1)
    show("erosion", erosion)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 3)
    show("dilation", dilation)

def main():
    erode_living('images/input/only_living.jpg')   
    # erode_living('images/input/sample1.jpg')   

if __name__ == '__main__':
    main()
