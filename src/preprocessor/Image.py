from ProcessState import ProcessState
from utils import fetch_name
from os.path import exists
import cv2


class Image:
    def __init__(self, name: str = None, original_image = None, state: ProcessState = ProcessState.default(), file_path: str = None, frame = None):
        if state.value == ProcessState.default() and not file_path and exists(file_path):
            raise Exception('Image cons: filepath must be valid for original image')
        self.frame = cv2.imread(file_path) if file_path else frame
        self.original_image = original_image
        if file_path:
            self.name = fetch_name(file_path) 
        elif name is None:
            self.name = self.original_image.name 
        else:
            self.name = name
        self.state = state
        self.file_path = file_path
        if not self.file_path and ((not self.original_image) or (self.frame is None or self.name is None)):
            raise Exception('Image cons: both filepath and (original_image or frame) cant be none.')        
        if not self.original_image: ## Huge danger zone. Please review
            self.original_image = self
