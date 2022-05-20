

from traceback import FrameSummary
import cv2
import numpy as np
 
def cca():
    image = cv2.imread('images/input/only_living_inpainted.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY)[1]
    print("thresh", thresh, thresh.shape)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)

    output = cv2.connectedComponentsWithStats(
	thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    # print("output", output, centroids)
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format( i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        print("area", area)
        if area < 5000:
            continue
        (cX, cY) = centroids[i]
        
        output = image.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (255, 255, 255), -1)
        
        componentMask = (labels == i).astype("uint8") * 255
        cv2.imshow("Connected Component", componentMask)
        cv2.waitKey(0)

    print("comp", componentMask)    
    # show our output image and connected component mask
    cv2.imshow("Output", output)
    cv2.imshow("Connected Component", componentMask)
    cv2.waitKey(0)

if __name__ == '__main__':
    cca()    
