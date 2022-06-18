from PreProcessor import PreProcessor



IMAGES_DIR = 'src/images'
INPUT_DIR = IMAGES_DIR + '/input/train'
OUTPUT_DIR = IMAGES_DIR + '/output'

DEBUG_IMGNAME_ALLOWLIST=[
    # '20220111_111647',
    # '20220111_111904',
    # '20220118_145609',
    # '20220111_111120',
    # '20220111_110953',
    # '20220111_111647'
    ]




def main():
    preprocessor_runtime = PreProcessor(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, debug=False, debug_imgname_allowlist=DEBUG_IMGNAME_ALLOWLIST, should_crop_image=True)
    preprocessor_runtime.preprocess_all()

if __name__ == '__main__':
    main()
