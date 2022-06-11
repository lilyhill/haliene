from typing import List
from Image import Image


class PreProcessor:
    def __init__(self, input_dir: str, output_dir: str, debug: bool = False, debug_imgname_allowlist: List[str] = []):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.debug = debug
        self.debug_imgname_allowlist = debug_imgname_allowlist
        self.mldata_subpath = 'ml_data'
        self.unlabelled_subpath = 'unlabelled'
        self.labelled_subpath = 'labelled'
        self.ccaed_subpath = 'ccaed'
    
    def should_debug(self, image_name):
        return self.debug and image_name in self.debug_imgname_allowlist

    def get_mldata_unlabelled_dir(self):
        return '/'.join([self.output_dir, self.mldata_subpath, self.unlabelled_subpath])
