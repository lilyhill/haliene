from Image import Image
from PreProcessor import PreProcessor
from common_ops import hue_separate, get_dustful
from morphological import reduce_noise_from_dustful_image, reduce_noise_from_hue_highlighted_image
from utils import show
import cv2


IMAGES_DIR = '../images'
INPUT_DIR = IMAGES_DIR + '/input'
OUTPUT_DIR = IMAGES_DIR + '/output'


# def preprocess_and_save(img_path, output_dir):


#     ## remove noise from asds_mask

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
    orig_image = Image(file_path=img_path)
    input_image = Image(file_path=img_path)
    input_image.original_image = orig_image ## Huge Red Flag -> just kill this thing. why is Image(file_path=img_path) repeated? THere is a leakage because if am doing self.original_image=self. latter was required so that hue_seperation can work with any image (and not just original_image) (see use of self.original_image there) and former is there so there to make it possible for hue_separation to work for original_image. i.e. former is because of latter.
    hue_separated_gray_image = hue_separate(preprocessor_runtime=preprocessor_runtime, image=input_image)
    dustful_image, hue_highligted, input_binary = get_dustful(preprocessor_runtime, hue_separated_gray_image)
    
    dustless_dustful_binary_image = reduce_noise_from_dustful_image(preprocessor_runtime=preprocessor_runtime, dustful_image = dustful_image)
    dustless_hue_highlighted_binary_image = reduce_noise_from_hue_highlighted_image(preprocessor_runtime=preprocessor_runtime, hue_highlighted=hue_highligted)
    
    print("from here")
    show("dustless_dustful_binary_image.frame", dustless_dustful_binary_image.frame)
    show("dustless_hue_highlighted_binary_image.frame", dustless_hue_highlighted_binary_image.frame)
    dirtless_image = cv2.bitwise_or(dustless_dustful_binary_image.frame, dustless_hue_highlighted_binary_image.frame, mask=input_binary.frame)

    show("dirtless_image", dirtless_image)
    return dirtless_image

    
if __name__ == '__main__':
    test()
