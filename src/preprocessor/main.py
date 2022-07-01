from PreProcessor import PreProcessor



IMAGES_DIR = 'src/images'
INPUT_DIR = IMAGES_DIR + '/input'
OUTPUT_DIR = IMAGES_DIR + '/output'

DEBUG_IMGNAME_ALLOWLIST=[
    ]
"""
small
469 53 54 
2814 71 62
2862 62 73
2501 67 59
2772 69 75
2556 58 71

medium
3848 76 84
3829 73 75
3829 73 75
4936 98 90
5058 88 70
4328 82 92
4270 80 81
3046 64 71


Big
6540 112 78
5033 82 79
7243 109 91
3718 48 131
5721 82 89
"""




def main():
    preprocessor_runtime = PreProcessor(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, debug=False, debug_imgname_allowlist=DEBUG_IMGNAME_ALLOWLIST, should_crop_image=True, train_or_test='test')
    preprocessor_runtime.preprocess_all()
    # preprocessor_runtime.fetch_input_for_ml()

if __name__ == '__main__':
    main()
