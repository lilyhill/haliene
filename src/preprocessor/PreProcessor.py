from typing import List
from Image import Image


class PreProcessor:
    def __init__(self, input_dir: str, output_dir: str, debug: bool = False, debug_imgname_allowlist: List[str] = []):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.debug = debug
        self.debug_imgname_allowlist = debug_imgname_allowlist
    
    def should_debug(self, image_name):
        return self.debug and image_name in self.debug_imgname_allowlist
