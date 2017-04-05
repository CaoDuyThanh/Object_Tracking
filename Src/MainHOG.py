from Utils.MOTDataHelper import *


# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'


#  GLOBAL VARIABLES
Dataset = None

# HOG Model


def LoadDataset():
    global Dataset
    Dataset = MOTDataHelper(DATASET_PATH)

def CreateModel():


    return 0

def TrainModel():
    return 0

def TestModel():
    return 0

if __name__ == '__main__':
    LoadDataset()
    CreateModel()
    TrainModel()
    TestModel()