import enum

class ProcessState(enum.Enum):
   ORIGINAL = 'original'
   HUE_SEPARATED = 'hue-separated'
   THRESOLDED = 'thresholded'

   def default():
      return ProcessState.ORIGINAL

