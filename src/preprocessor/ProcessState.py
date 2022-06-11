import enum

class ProcessState(enum.Enum):
   ORIGINAL = 'original'
   HUE_SEPARATED = 'hue-separated'
   HUE_HIGHLIGHTED = 'hue-highlighted' ## hue highlighted is the product after binary thresholding ofr hue-seperated
   THRESOLDED = 'thresholded'
   DUSTFUL = 'dustful'
   ERODED = 'eroded'
   DILATED = 'dilated'
   MORPHED = 'morphed'
   BOUNDING_BOXED = 'bounding-boxed'
   DIRTLESS = 'dirtless'
   UNNAMED = 'unamed'

   def default():
      return ProcessState.ORIGINAL

