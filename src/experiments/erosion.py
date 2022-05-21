import cv2

def remove_living():
    image = cv2.imread('images/input/sample1.jpg')
    only_living_mask = cv2.cvtColor(cv2.imread('images/input/only_living_mask.jpg'), cv2.COLOR_BGR2GRAY)
    only_nonliving_mask = cv2.bitwise_not(only_living_mask)
    print("image", image.shape)
    print("only_nonliving_mask", only_nonliving_mask.shape)
    image_with_gaps_left_by_living_sperms = cv2.bitwise_and(image, image, mask=only_nonliving_mask)
    cv2.imwrite("images/output/image_with_gaps_left_by_living_sperms.jpg", image_with_gaps_left_by_living_sperms)

    # Above image still has black holes left after removing our living sperms.
    image_without_living = cv2.inpaint(image_with_gaps_left_by_living_sperms, only_living_mask, 5, flags=cv2.INPAINT_TELEA)
    cv2.imwrite("images/output/without_living.jpg", image_without_living)
    cv2.imshow("image_without_living", image_without_living)
    cv2.waitKey(0)



def main():
    remove_living()    

if __name__ == '__main__':
    main()
