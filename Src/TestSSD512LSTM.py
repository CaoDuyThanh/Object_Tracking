import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from FeaturesExtraction.SSD512FeaExtraction import *
from Utils.MOTDataHelper import *
from Models.LSTM.LSTMTrackingModel import *
from Utils.DefaultBox import *
from Utils.BBoxHelper import *
from Utils.FileHelper import *

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# TRAINING HYPER PARAMETER
NUM_EPOCH         = 10
DISPLAY_FREQUENCY = 50
SAVE_FREQUENCY    = 1000

# LSTM NETWORK CONFIG
BATCH_SIZE        = 1
NUM_TRUNCATE      = 1
NUM_HIDDEN        = 512
INPUTS_SIZE       = [5461 + 5461]
OUTPUTS_SIZE      = [5461]

# BOUNDING BOX HYPER
ALPHA = 0.6
BETA  = 1.4

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/SSD512/LSTM_SSD_Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 0
START_ITERATION = 5800

#  GLOBAL VARIABLES
Dataset           = None
FeatureFactory    = None
DefaultBboxs      = None
LSTMModel         = None


########################################################################################################################
#                                                                                                                      #
#    LOAD DATASET SESSIONS                                                                                             #
#                                                                                                                      #
########################################################################################################################
def LoadDataset():
    global Dataset
    if DATASET_SOURCE == 'MOT':
        Dataset = MOTDataHelper(DATASET_PATH)


########################################################################################################################
#                                                                                                                      #
#    CREATE SSD EXTRACT FEATURES FACTORY                                                                               #
#                                                                                                                      #
########################################################################################################################
def CreateSSDExtractFactory():
    global FeatureFactory, DefaultBboxs
    FeatureFactory = SSD512FeaExtraction()
    FeatureFactory.LoadCaffeModel('../Models/SSD_512x512/VOC0712/deploy.prototxt',
                                  '../Models/SSD_512x512/VOC0712/VGG_coco_SSD_512x512_iter_360000.caffemodel')
    DefaultBboxs = FeatureFactory.GetDefaultBbox(imageWidth=512,
                                                 sMin=10,
                                                 sMax=90,
                                                 layerSizes=[(64, 64),
                                                             (32, 32),
                                                             (16, 16),
                                                             (8, 8),
                                                             (4, 4),
                                                             (2, 2),
                                                             (1, 1)],
                                                 numBoxs=[6, 6, 6, 6, 6, 6, 6],
                                                 offset=0.5,
                                                 steps=[8, 16, 32, 64, 128, 256, 512])


########################################################################################################################
#                                                                                                                      #
#    CREATE LSTM MODEL FOR TRACKING OBJECTS                                                                            #
#                                                                                                                      #
########################################################################################################################
def CreateLSTMModel():
    global LSTMModel, FeatureFactory
    LSTMModel = LSTMTrackingModel(featureFactory = FeatureFactory,
                                  batchSize      = BATCH_SIZE,
                                  numTruncate    = NUM_TRUNCATE,
                                  numHidden      = NUM_HIDDEN,
                                  inputsSize     = INPUTS_SIZE,
                                  outputsSize    = OUTPUTS_SIZE)

########################################################################################################################
#                                                                                                                      #
#    UTILITIES (MANY DIRTY CODES)                                                                                      #
#                                                                                                                      #
########################################################################################################################
def CreateHeatmapSequence(defaultBboxs, groundTruths):
    heatmapSequence = []
    for idx, groundTruth in enumerate(groundTruths):
        heatmap = CreateHeatmapF(defaultBboxs, groundTruth)
        heatmapSequence.append(heatmap)

    # Convert to numpy array
    heatmapSequence = numpy.asarray(heatmapSequence, dtype='float32')
    return heatmapSequence

def Compare(ass,bss):
    for (a,b) in zip(ass,bss):
        if a != b:
            return False
    return True

def CreateHeatmap(defaultBboxs, groundTruth):
    heatmap = numpy.zeros((defaultBboxs.shape[0], 1), dtype = 'float32')
    for bboxIdx, dfbbox in enumerate(defaultBboxs):
        check = False
        for archorboxIdx, archorBox in enumerate(dfbbox):
            minInterestBox, maxInterestBox = InterestBox(archorBox, groundTruth, ALPHA, BETA)
            if minInterestBox >= 0.25:
                check = True
                break
        if check == True:
            heatmap[bboxIdx] = 1
    return heatmap

def CreateHeatmapF(defaultBboxs, groundTruth):
    numPos        = defaultBboxs.shape[0]
    numBboxPerPos = defaultBboxs.shape[1]
    sizePerBbox   = defaultBboxs.shape[2]

    defaultBboxs = defaultBboxs.reshape((numPos * numBboxPerPos, sizePerBbox))
    zeros        = numpy.zeros((numPos * numBboxPerPos))
    interX       = numpy.maximum(zeros,
                                 numpy.minimum(
                                     defaultBboxs[:, 0] + defaultBboxs[:, 2] / 2,
                                     groundTruth [0] + groundTruth [2] / 2
                                 ) -
                                 numpy.maximum(
                                     defaultBboxs[:, 0] - defaultBboxs[:, 2] / 2,
                                     groundTruth [0] - groundTruth [2] / 2
                                 ))
    interY       = numpy.maximum(zeros,
                                 numpy.minimum(
                                     defaultBboxs[:, 1] + defaultBboxs[:, 3] / 2,
                                     groundTruth [1] + groundTruth [3] / 2
                                 ) -
                                 numpy.maximum(
                                     defaultBboxs[:, 1] - defaultBboxs[:, 3] / 2,
                                     groundTruth [1] - groundTruth [3] / 2
                                 ))
    iterArea = interX * interY

    area1 = defaultBboxs[:, 2] * defaultBboxs[:, 3]
    area2 = groundTruth [2] * groundTruth [3]

    ratio1 = iterArea / area1
    ratio2 = iterArea / area2

    heatmap = numpy.zeros((numPos * numBboxPerPos,), dtype = 'float32')
    index = numpy.where(ratio1 > 0.25)
    heatmap[index] = 1
    heatmap = heatmap.reshape((numPos, numBboxPerPos))
    heatmap = heatmap.sum(axis = 1)
    index = numpy.where(heatmap > 0)
    heatmap = numpy.zeros((numPos,), dtype='float32')
    heatmap[index] = 1
    heatmap = heatmap.reshape((numPos, 1))
    return heatmap

########################################################################################################################
#                                                                                                                      #
#    TEST LSTM MODEL........................                                                                           #
#                                                                                                                      #
########################################################################################################################
def TestModel():
    global Dataset, LSTMModel, FeatureFactory, DefaultBboxs

    # Create startStateS | startStateC
    startStateS = numpy.zeros((BATCH_SIZE, LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    startStateC = numpy.zeros((BATCH_SIZE, LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')

    # Plot training cost
    plt.ion()
    fig, ax = plt.subplots(2, 4)
    ab = None

    # Load model
    if CheckFileExist(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION),
                      throwError=False) == True:
        file = open(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION))
        LSTMModel.LoadModel(file)
        file.close()
        print ('Load model !')

    S = startStateS
    C = startStateC

    # Test each Folder
    Dataset.DataOpts['data_phase'] = 'train'
    allFolderNames = Dataset.GetAllFolderNames()
    for folderName in allFolderNames:
        folderName = 'MOT16-11'
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_folder_type'] = 'gt'
        imsPath, bboxs = Dataset.GetRandomBbox()

        S = startStateS; C = startStateC

        inputBatch    = numpy.zeros((BATCH_SIZE * NUM_TRUNCATE, 3, 512, 512), dtype='float32')
        heatmapXBatch = numpy.zeros((BATCH_SIZE, NUM_TRUNCATE, 5461, 1), dtype='float32')
        id = 1
        while id < imsPath.__len__():
            imPath = imsPath[id]

            # Extract feature and prepare bounding box before training....
            inputSequence = ReadImages(imsPath = [imPath], batchSize = NUM_TRUNCATE)   # Extract batch features
            if id == 1:
                heatmapSequence = CreateHeatmapSequence(DefaultBboxs, [bboxs])
            else:
                heatmapSequence = heatmap.reshape(1, 5461, 1)

            inputBatch[0, :, :, :]    = inputSequence
            heatmapXBatch[0, 0, :, :] = heatmapSequence

            heatmap, S, C = LSTMModel.NextState(inputBatch, heatmapXBatch, S, C)
            heatmapIdxP = numpy.where(heatmap > 0.50)
            heatmapIdxN = numpy.where(heatmap <= 0.50)
            heatmap[heatmapIdxP] = 1
            heatmap[heatmapIdxN] = 0
            print (numpy.sum(heatmap))

            S = numpy.asarray([S], dtype = 'float32')
            C = numpy.asarray([C], dtype = 'float32')

            # Get data from heatmap
            idx = numpy.asarray([0, 64 * 64, 32 * 32, 16 * 16, 8 * 8, 4 * 4, 2 * 2, 1 * 1], dtype = 'int32')
            idx = numpy.cumsum(idx)
            win64 = heatmap[idx[0] : idx[1]]; win64 = win64.reshape((64, 64))
            win32 = heatmap[idx[1] : idx[2]]; win32 = win32.reshape((32, 32))
            win16 = heatmap[idx[2] : idx[3]]; win16 = win16.reshape((16, 16))
            win8  = heatmap[idx[3] : idx[4]]; win8  = win8.reshape((8, 8))
            win4  = heatmap[idx[4] : idx[5]]; win4  = win4.reshape((4, 4))
            win2  = heatmap[idx[5] : idx[6]]; win2  = win2.reshape((2, 2))
            win1  = heatmap[idx[6] : idx[7]]; win1  = win1.reshape((1, 1))

            rawIm = cv2.imread(imPath)
            h, w, _ = rawIm.shape

            if ab == None:
                ab1 = ax[0, 0].imshow(rawIm)
                ab2 = ax[0, 1].imshow(win64)
                ab3 = ax[0, 2].imshow(win32)
                ab4 = ax[0, 3].imshow(win16)
                ab5 = ax[1, 0].imshow(win8)
                ab6 = ax[1, 1].imshow(win4)
                ab7 = ax[1, 2].imshow(win2)
                ab8 = ax[1, 3].imshow(win1)
                plt.pause(5.0)
            else:
                ab1.set_data(rawIm)
                ab2.set_data(win64)
                ab3.set_data(win32)
                ab4.set_data(win16)
                ab5.set_data(win8)
                ab6.set_data(win4)
                ab7.set_data(win2)
                ab8.set_data(win1)
            plt.show()
            plt.axis('off')
            plt.pause(0.00001)

            id += 1

def DrawHeatmap(imsPath, heatmaps):
    fig, ax = plt.subplots(2, 4)
    ab = None

    for heatmap in heatmaps:
        idx = numpy.asarray([0, 64 * 64, 32 * 32, 16 * 16, 8 * 8, 4 * 4, 2 * 2, 1 * 1], dtype='int32')
        idx = numpy.cumsum(idx)
        win64 = heatmap[idx[0]: idx[1]];
        win64 = win64.reshape((64, 64))
        win32 = heatmap[idx[1]: idx[2]];
        win32 = win32.reshape((32, 32))
        win16 = heatmap[idx[2]: idx[3]];
        win16 = win16.reshape((16, 16))
        win8 = heatmap[idx[3]: idx[4]];
        win8 = win8.reshape((8, 8))
        win4 = heatmap[idx[4]: idx[5]];
        win4 = win4.reshape((4, 4))
        win2 = heatmap[idx[5]: idx[6]];
        win2 = win2.reshape((2, 2))
        win1 = heatmap[idx[6]: idx[7]];
        win1 = win1.reshape((1, 1))

        if ab == None:
            ab2 = ax[0, 1].imshow(win64)
            ab3 = ax[0, 2].imshow(win32)
            ab4 = ax[0, 3].imshow(win16)
            ab5 = ax[1, 0].imshow(win8)
            ab6 = ax[1, 1].imshow(win4)
            ab7 = ax[1, 2].imshow(win2)
            ab8 = ax[1, 3].imshow(win1)
        else:
            ab2.set_data(win64)
            ab3.set_data(win32)
            ab4.set_data(win16)
            ab5.set_data(win8)
            ab6.set_data(win4)
            ab7.set_data(win2)
            ab8.set_data(win1)
        plt.show()
        plt.axis('off')
        plt.pause(0.00001)
        # rect.remove()


def Draw(imsPath, bboxs):
    fig, ax = plt.subplots(1)
    ab = None

    for imPath, bbox in zip(imsPath, bboxs):
        rawIm = cv2.imread(imPath)

        if ab == None:
            ab = ax.imshow(rawIm)
        else:
            ab.set_data(rawIm)

        cx = bbox[0]
        cy = bbox[1]
        w  = bbox[2]
        h  = bbox[3]

        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

        h, w, _ = rawIm.shape
        rect = patches.Rectangle((box[0] * w, box[1] * h), (box[2] - box[0]) * w,
                                 (box[3] - box[1]) * h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()
        plt.axis('off')

        plt.pause(5)

        rect.remove()


if __name__ == '__main__':
    LoadDataset()
    CreateSSDExtractFactory()
    CreateLSTMModel()
    TestModel()