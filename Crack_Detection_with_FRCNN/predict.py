from PIL import Image
from numpy import *
import faster_rcnn_pb2 as fr

class crack_detection:
    def __init__(self,filename):
        self.filename =filename

    def predictiondogcat(self):
        image_path = self.filename
        fr.main(image_path)
