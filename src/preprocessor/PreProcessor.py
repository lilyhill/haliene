from typing import List
from utils import show, resize_and_grayscale
from Image import Image
from common_ops import hue_separate, get_dustful, cca
from morphological import reduce_noise_from_dustful_image, reduce_noise_from_hue_highlighted_image
from ProcessState import ProcessState
import cv2
import os
import numpy as np
import random
random.seed(0)

all_input_images_for_ml = {
    'degraded': [],
    'dust': [],
    'big_halo': [],
    'medium_halo': [],
    'small_halo': []
}
label_for_input_images_for_ml = {

}

label_list = [
    'degraded',
    'dust',
    'big_halo',
    'medium_halo',
    'small_halo'
]    

class PreProcessor:
    def __init__(self, input_dir: str, output_dir: str, debug: bool = False, debug_imgname_allowlist: List[str] = [], should_crop_image = True, train_or_test = 'train'):
        self.input_dir = input_dir + '/' + train_or_test
        self.output_dir = output_dir
        self.debug = debug
        self.debug_imgname_allowlist = debug_imgname_allowlist
        self.mldata_subpath = 'ml_data' + '/' + train_or_test
        self.unlabelled_subpath = 'unlabelled'
        self.labelled_subpath = 'labelled'
        self.ccaed_subpath = 'ccaed'
        self.should_crop_image = should_crop_image
        self.train_or_test = train_or_test
    
    def fetch_input_for_ml(self):
        '''
        read all the folders inside ml_data/train and ml_data/test and use them to label in five classes
        '''
        output_dir = os.path.join(self.output_dir, self.mldata_subpath, 'labelled')
        for dir in os.listdir(output_dir):
            image_named_dir = os.path.join(output_dir, dir)
            if not os.path.isdir(image_named_dir):
                continue
            for dir_ in os.listdir(image_named_dir):
                size_dir = os.path.join(image_named_dir, dir_)
                if not os.path.isdir(size_dir):
                    continue
                for dir__ in os.listdir(size_dir):
                    class_dir = os.path.join(size_dir, dir__)
                    if not os.path.isdir(class_dir):
                        continue
                    for image_file in os.listdir(class_dir):
                        if image_file == '.DS_Store':
                            continue
                        image_file_path = os.path.join(class_dir, image_file)
                        print("image_file_path", image_file_path)
                        image_frame = resize_and_grayscale(image_file_path)
                        print("image_frame", image_frame.shape, image_file_path)
                        if not os.path.isfile(image_file_path):
                            continue
                        if dir__ == 'degraded':
                            all_input_images_for_ml['degraded'].append(image_frame)
                        elif dir__ == 'dust':
                            all_input_images_for_ml['dust'].append(image_frame)
                        elif dir__ == 'nondegraded':
                            if dir_ == 'big':
                                all_input_images_for_ml['big_halo'].append(image_frame)
                            elif dir_ == 'small':
                                all_input_images_for_ml['small_halo'].append(image_frame)
                            elif dir_ == 'medium':
                                all_input_images_for_ml['medium_halo'].append(image_frame)

        final_list = []
        for key in all_input_images_for_ml:
            label_ind = label_list.index(key)
            all_input_images_for_ml[key] = [(img, label_ind) for img in all_input_images_for_ml[key]]
            print('all_input_images_for_ml[key]', len(all_input_images_for_ml[key]))
            final_list.append(all_input_images_for_ml[key])
        
        random.shuffle(final_list)
        # verify shuffle
        # divide it in two parts
        # next feed it to ml
        

                    
        

            
        

    def should_debug(self, image_name):
        return self.debug and image_name in self.debug_imgname_allowlist

    def get_mldata_unlabelled_dir(self):
        return '/'.join([self.output_dir, self.mldata_subpath, self.unlabelled_subpath])

    def preprocess_all(self):
        input_images = os.listdir(self.input_dir) if not self.debug else [f + '.jpg' for f in self.debug_imgname_allowlist]
        for f in input_images:
            img_path = os.path.join(self.input_dir, f)
            filepath, ext = os.path.splitext(img_path)
            if os.path.isfile(img_path) and ext == '.jpg':
                try:
                    self.preprocess_single(img_path)                
                except Exception as e:
                    print("eee " + img_path + ' ' + str(e))

    def preprocess_single(self, img_path: str):
        orig_image = Image(file_path=img_path)
        if self.should_debug(orig_image.name):
            show('original ' + orig_image.name, orig_image.frame)
        input_image = Image(file_path=img_path)
        input_image.original_image = orig_image ## Huge Red Flag -> just kill this thing. why is Image(file_path=img_path) repeated? THere is a leakage because if am doing self.original_image=self. latter was required so that hue_seperation can work with any image (and not just original_image) (see use of self.original_image there) and former is there so there to make it possible for hue_separation to work for original_image. i.e. former is because of latter.
        hue_separated_gray_image = hue_separate(preprocessor_runtime=self, image=input_image)
        dustful_image, hue_highligted, input_binary = get_dustful(preprocessor_runtime=self, image = hue_separated_gray_image)
        
        dustless_dustful_binary_image = reduce_noise_from_dustful_image(preprocessor_runtime=self, dustful_image = dustful_image)
        dustless_hue_highlighted_binary_image = reduce_noise_from_hue_highlighted_image(preprocessor_runtime=self, hue_highlighted=hue_highligted)

        dirtless_image = Image(
            frame = cv2.bitwise_or(dustless_dustful_binary_image.frame, dustless_hue_highlighted_binary_image.frame, mask=input_binary.frame),
            state = ProcessState.DIRTLESS,
            original_image = orig_image
        )

        ccaed_img = cca(preprocessor_runtime = self, binary_image = dirtless_image)
        ccaed_img_path = self.output_dir + '/' + self.ccaed_subpath + '/' + 'ccaed_' + dirtless_image.name + '.jpg'
        if self.should_debug(ccaed_img.name):
            show('cca ' + ccaed_img.name, ccaed_img.frame)
        
        cv2.imwrite(ccaed_img_path, ccaed_img.frame)
        
        return ccaed_img_path