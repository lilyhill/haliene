def erode(dustful_img, kernelsize = 8, iterations = 2):
    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    return cv2.erode(dustful_img, kernel, iterations = iterations)

def dilate(eroded, kernelsize = 3, iterations = 10):
    kernel = np.ones((kernelsize, kernelsize),np.uint8)
    return cv2.dilate(eroded,kernel,iterations = iterations)