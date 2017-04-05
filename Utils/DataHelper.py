import cv2
import numpy

class DatasetHelper:
    # ------------------------  LOAD TRAIN | VALID | TEST FILES--------------------------------------------
    def loadTrainFile(self): raise NotImplementedError

    def loadValidFile(self): raise NotImplementedError

    def loadTestFile(self): raise NotImplementedError

    # -----------------------------------------------------------------------------------------------------
    def NextTrainBatch(self): raise NotImplementedError

    def NextValidBatch(self): raise NotImplementedError

    def NextTestBatch(self, batchSize): raise NotImplementedError
