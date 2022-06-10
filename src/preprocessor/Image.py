from ProcessState import ProcessState
from utils import fetch_name
import cv2

class Image:
    def __init__(self,  original_image_path: str, frame = None, name: str = None, original_frame = None, state: ProcessState = ProcessState.default()):
        if not original_image_path and not frame:
            raise Exception("both original_image_path and frame can't be empty")
        self.frame = frame if frame else cv2.imread(original_image_path)
        self.name = fetch_name(original_image_path) if not name else name
        self.state = state
        self.original_frame = original_frame if original_frame else frame
        self.original_image_path = original_image_path
