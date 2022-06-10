from Image import Image
from PreProcessor import PreProcessor
from common_ops import hue_separate, get_dustful
from morphological import reduce_noise_from_dustful_image, reduce_noise_from_hue_highlighted_image
import cv2


IMAGES_DIR = '../images'
INPUT_DIR = IMAGES_DIR + '/input'
OUTPUT_DIR = IMAGES_DIR + '/output'

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

    dirtless_image = cv2.bitwise_or(dustless_dustful_binary_image.frame, dustless_hue_highlighted_binary_image.frame, mask=input_binary.frame)

    return dirtless_image

    
if __name__ == '__main__':
    test()
