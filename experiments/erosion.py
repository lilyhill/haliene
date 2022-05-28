import cv2

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def remove_living():
    image = cv2.imread('images/input/sample1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show("gray", gray)
    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    show("thresh", thresh)
    only_living_mask = cv2.threshold(cv2.cvtColor(cv2.imread('images/input/only_living.jpg'), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_TOZERO)[1]
    show("only_living_mask", only_living_mask)

    only_living_mask2 = cv2.threshold(only_living_mask, 0, 255, cv2.THRESH_BINARY)[1]
    show("only_living_mask2", only_living_mask2)

    no_living_thresh_with_black_gaps = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(only_living_mask))

    show("no_living_thresh_with_black_gaps", no_living_thresh_with_black_gaps)

    no_living_thresh = cv2.bitwise_or(no_living_thresh_with_black_gaps, only_living_mask2, only_living_mask)
    show("no_living_threshold", no_living_thresh)
    
def main():
    remove_living()    

if __name__ == '__main__':
    main()
