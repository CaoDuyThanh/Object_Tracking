import cv2
import numpy

class DatasetHelper:
    def __init__(self):
        self.DataOpts = {}

        # Default setting for retrieve data
        self.DataOpts['data_phase']       = 'train'     # Setting 'train' for training
                                                        #         'test' for testing
        self.DataOpts['data_folder_name'] = ''
        self.DataOpts['data_folder_type'] = 'gt'        # Setting 'gt' for tracking
                                                        # Setting 'det' for detection
        self.DataOpts['data_object_id']   = '0'

    # ------------------------  LOAD TRAIN | VALID | TEST FILES--------------------------------------------
    def loadTrainFile(self): raise NotImplementedError

    def loadValidFile(self): raise NotImplementedError

    def loadTestFile(self): raise NotImplementedError

    # -----------------------------------------------------------------------------------------------------
    def NextTrainBatch(self): raise NotImplementedError

    def NextValidBatch(self): raise NotImplementedError

    def NextTestBatch(self, batchSize): raise NotImplementedError
