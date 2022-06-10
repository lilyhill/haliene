
import cv2
import os


from Image import Image
from PreProcessor import PreProcessor
from common_ops import hue_separate, get_full_img_binary

IMAGES_DIR = '../images'
INPUT_DIR = IMAGES_DIR + '/input'
OUTPUT_DIR = IMAGES_DIR + '/output'


# def preprocess_and_save(img_path, output_dir):
#     img_frame = cv2.imread(img_path)
#     gray = hue_separate(img_frame)
#     full_img_binary_img = get_full_img_binary(gray)
#     # full_img_binary_img = cv2.cvtColor(full_img_binary_img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(full_img_binary, full_img_binary_img)
#     asds_mask = cv2.cvtColor(asds_mask, cv2.COLOR_BGR2GRAY)
#     # show("asds_mask", asds_mask)
#     dustful = cv2.bitwise_or(full_img_binary_img, asds_mask, asds_mask)
#     # show("dustful", dustful)
#     dustful = cv2.bitwise_not(dustful)


#     # infill = inpainting(bssdds, asds_mask)

#     ## morphological operations on bdds
#     eroded = erode(dustful)
#     # show("erosion", eroded)

#     bdds_dilated = dilate(eroded)
#     # show("dilation", bdds_dilated)


#     ## remove noise from asds_mask
#     asds_mask = cv2.imread('../images/output/asds_thresh.jpg')
#     # show("asds_mask", asds_mask)
#     eroded = erode(asds_mask, kernelsize=6, iterations=2)
#     # show("erosion1", eroded)
#     cv2.imwrite("../images/output/asds_mask_eroded.jpg", eroded)

#     dilated = dilate(eroded, kernelsize=6, iterations=8)
#     # show("dilation1", dilated)
#     cv2.imwrite("../images/output/asds_mask_dilated.jpg", dilated)

#     asds_mask_clean = cv2.bitwise_and(asds_mask, dilated, asds_mask)
#     # show("asds_mask_clean", asds_mask_clean)
#     cv2.imwrite("../images/output/asds_mask_clean.jpg", asds_mask_clean)

#     ## club asds_mask_clean and bdds and form bddsas mask 
#     print("asds", asds_mask_clean.shape, bdds_dilated.shape)
#     asds_mask_clean_gray = cv2.cvtColor(asds_mask_clean, cv2.COLOR_BGR2GRAY)
#     bddsas_mask = cv2.bitwise_or(asds_mask_clean_gray, bdds_dilated, bdds_dilated)
#     # show("bddsas_mask", bddsas_mask)
#     cv2.imwrite("../images/output/bddsas_mask.jpg", bddsas_mask)


#     ## use bddsas mask to create final dirtless image.
#     full_img_binary_img_thresh = cv2.bitwise_not(cv2.threshold(full_img_binary_img, 100, 255,
# 	cv2.THRESH_BINARY)[1])
#     # show("full_img_binary_img_thresh", full_img_binary_img_thresh)
#     dirtless = cv2.bitwise_and(full_img_binary_img_thresh, bddsas_mask, bddsas_mask)
#     # show("dirtless", dirtless)
#     cv2.imwrite("../images/output/dirtless.jpg", dirtless)

#     ## apply cca and form bounding boxes around blobs. 
#     ccaed_img = cca(dirtless)
#     cv2.imwrite("../images/output/ccaed_img.jpg", ccaed_img)

def test_hue_separation(preprocessor_runtime, image):
    blobby_image = hue_separate(preprocessor_runtime=preprocessor_runtime, image=image)
    print(type(blobby_image), blobby_image)
    return blobby_image





def test():
    preprocessor_runtime = PreProcessor(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, debug=True, debug_imgname_allowlist=['sample1'])
    img_path = '/Users/nilinswap/forgit/others/haliene/src/images/input/sample1.jpg'
    image = Image(file_path=img_path)
    blobby_image = hue_separate(preprocessor_runtime=preprocessor_runtime, image=image)
    dirtsomething = get_full_img_binary(preprocessor_runtime, blobby_image)
    return dirtsomething

    
if __name__ == '__main__':
    test()