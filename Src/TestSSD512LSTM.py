import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from FeaturesExtraction.SSD512FeaExtraction import *
from Utils.MOTDataHelper import *
from Models.LSTM.LSTMTrackingModel import *
from Utils.DefaultBox import *
from Utils.BBoxHelper import *

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
NUM_TRUNCATE      = 1
NUM_HIDDEN        = 512
INPUTS_SIZE       = [256]
OUTPUTS_SIZE      = [6, 24]

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/SSD512/LSTM_SSD_Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 1
START_ITERATION = 8550

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
    FeatureFactory = SSD512FeaExtraction(batchSize = NUM_TRUNCATE)
    FeatureFactory.LoadCaffeModel('../Models/SSD_512x512/VOC0712/deploy.prototxt',
                                  '../Models/SSD_512x512/VOC0712/VGG_coco_SSD_512x512_iter_360000.caffemodel')
    FeatureFactory.LoadEncodeLayers('../Preprocessing/SSD512/ssd512_conv4_3_norm_encode.pkl',
                                    '../Preprocessing/SSD512/ssd512_fc7_encode.pkl',
                                    '../Preprocessing/SSD512/ssd512_conv6_2_encode.pkl')
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
    global LSTMModel
    LSTMModel = LSTMTrackingModel(numTruncate = NUM_TRUNCATE,
                                  numHidden   = NUM_HIDDEN,
                                  inputsSize  = INPUTS_SIZE,
                                  outputsSize = OUTPUTS_SIZE)

########################################################################################################################
#                                                                                                                      #
#    UTILITIES (MANY DIRTY CODES)                                                                                      #
#                                                                                                                      #
########################################################################################################################
def CompareBboxs(defaultBboxs, groundTruths):
    preds = []
    gts   = []
    for idx, groundTruth in enumerate(groundTruths):
        pred, gt = CreateOutput(defaultBboxs, groundTruth)
        preds.append(pred)
        gts.append(gt)

    # Convert to numpy array
    preds = numpy.asarray(preds, dtype='float32')
    gts   = numpy.asarray(gts, dtype='float32')
    return preds, gts


def CreateOutput(defaultBboxs, groundTruth):
    pred = numpy.zeros((defaultBboxs.shape[0], defaultBboxs.shape[1], 1),
                        dtype = 'float32')
    gt   = numpy.zeros(defaultBboxs.shape,
                        dtype = 'float32')
    for bboxIdx, dfbbox in enumerate(defaultBboxs):
        for archorboxIdx, archorBox in enumerate(dfbbox):
            minInterestBox, maxInterestBox = InterestBox(archorBox, groundTruth)
            if minInterestBox >= 0.25 and maxInterestBox >= 0.5:
                pred[bboxIdx][archorboxIdx] = 1
                gt[bboxIdx][archorboxIdx]   = [(groundTruth[0] - archorBox[0]) / archorBox[2] / 0.1,
                                               (groundTruth[1] - archorBox[1]) / archorBox[3] / 0.1,
                                                math.log(groundTruth[2] / archorBox[2]) / 0.2,
                                                math.log(groundTruth[3] / archorBox[3]) / 0.2]
    return [pred, gt]


def GetFeatures(features, preds, defaultBboxs):
    featuresgts = []
    temp       = []
    for (feature, pred) in zip(features, preds):
        ftgt = []
        for idx in range(feature.shape[0]):
            p = numpy.max(pred[idx])

            if p == 1:
                ftgt.append(feature[idx])
                temp.append(defaultBboxs[idx])
        featuresgts.append(ftgt)
    return featuresgts, temp


def GetRandomFeatures(features, bboxs):
    featuresgt = numpy.zeros((features.__len__(), features[0][0].shape[0]), dtype='float32')
    temp = []
    for idx in range(features.__len__()):
        randNum = numpy.random.randint(features[idx].__len__())
        featuresgt[idx] = features[idx][randNum]
        temp.append(bboxs[randNum])

    return featuresgt, temp


def getBboxs(features, defaultBboxs, predConfs, predBboxs, threshold):
    bestBoxes = []
    bestConf = 0.0
    bestFeature = None
    for idx, predConf in enumerate(predConfs):
        for idx1, pred in enumerate(predConf):
            if pred >= threshold:
                if bestConf < pred:
                    bestConf = pred
                    bestFeature = features[idx]

                archorBox = defaultBboxs[idx][idx1]
                box       = predBboxs[idx][idx1]

                cx = archorBox[0]
                cy = archorBox[1]
                w  = archorBox[2]
                h  = archorBox[3]

                offsetXmin = box[0]
                offsetYmin = box[1]
                offsetXmax = box[2]
                offsetYmax = box[3]

                cx = offsetXmin / 0.1 * w + cx
                cy = offsetYmin / 0.1 * h + cy
                w = math.exp(offsetXmax / 0.2) * w
                h = math.exp(offsetYmax / 0.2) * h

                xmin = cx - w / 2.
                ymin = cy - h / 2.
                xmax = cx + w / 2.
                ymax = cy + h / 2.

                xmin = min(max(xmin, 0), 1)
                ymin = min(max(ymin, 0), 1)
                xmax = min(max(xmax, 0), 1)
                ymax = min(max(ymax, 0), 1)

                bestBoxes.append([xmin, ymin, xmax, ymax])
    return bestBoxes, bestFeature



########################################################################################################################
#                                                                                                                      #
#    TEST LSTM MODEL........................                                                                           #
#                                                                                                                      #
########################################################################################################################
def TestModel():
    global Dataset, LSTMModel, FeatureFactory, DefaultBboxs

    # Create startStateS | startStateC
    startStateS = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    startStateC = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    # startStateS = None
    # startStateC = None

    # Plot training cost
    plt.ion()
    fig, ax = plt.subplots(1)
    ab = None

    # Load model
    file = open(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION))
    LSTMModel.LoadModel(file)
    file.close()
    print ('Load model !')

    # Train each folder in train folder
    defaultBboxs = None
    iter = START_ITERATION
    costs = []

    # Test each Folder
    Dataset.DataOpts['data_phase'] = 'test'
    allFolderNames = Dataset.GetAllFolderNames()
    for folderName in allFolderNames:
        folderName = 'MOT16-07'
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_folder_type'] = 'det'
        imsPath, bboxs = Dataset.GetRandomBbox()

        S = startStateS
        C = startStateC

        while True:
            # Extract feature and prepare bounding box before training....
            batchFeatures          = FeatureFactory.ExtractFeature(imsPath = [imsPath[1]])   # Extract batch features
            batchPreds, batchBboxs = CompareBboxs(DefaultBboxs, [bboxs])

            inputBatchFeatures   = batchFeatures[0 : NUM_TRUNCATE]
            outputPreds          = batchPreds[0 : ]

            inputBatchFeatureGts, temp = GetFeatures(inputBatchFeatures, outputPreds, DefaultBboxs)
            inputBatchFeatureGt, temp  = GetRandomFeatures(inputBatchFeatureGts, temp)

            # Draw([imsPath[1],imsPath[1],imsPath[1],imsPath[1],imsPath[1],imsPath[1]], temp[0])

            S, C = LSTMModel.NextState(inputBatchFeatureGt, S, C)

            imPath = imsPath[1]
            rawIm = cv2.imread(imPath)
            h, w, _ = rawIm.shape

            if ab == None:
                ab = ax.imshow(rawIm)
            else:
                ab.set_data(rawIm)

            listRect = []
            box = bboxs
            rect = patches.Rectangle(((box[0] - box[2] / 2 ) * w, (box[1] - box[3] / 2)  * h),
                                       box[2] * w,
                                       box[3] * h, linewidth=1, edgecolor='b', facecolor='none')
            listRect.append(rect)
            # Add the patch to the Axes
            ax.add_patch(rect)

            plt.show()
            plt.axis('off')
            plt.pause(5)
            for rec in listRect:
                rec.remove()

            id = 2
            while id < imsPath.__len__():
                imPath = imsPath[id]
                batchFeatures = FeatureFactory.ExtractFeature(imsPath = [imPath])   # Extract batch features

                predConfs, predBboxs = LSTMModel.PredFunc(batchFeatures, S, C)

                predBboxs = predBboxs.reshape((predBboxs.shape[0], 6, 4))
                bestBboxs, bestFeature = getBboxs(batchFeatures[0], DefaultBboxs, predConfs, predBboxs, 0.3)

                if bestFeature is not None:
                    bestFeature = numpy.asarray([bestFeature], dtype = 'float32')
                    S, C        = LSTMModel.NextState(bestFeature, S, C)

                rawIm = cv2.imread(imPath)
                h, w, _ = rawIm.shape

                if ab == None:
                    ab = ax.imshow(rawIm)
                else:
                    ab.set_data(rawIm)

                listRect = []
                for box in bestBboxs:
                    rect = patches.Rectangle((box[0] * w, box[1] * h), (box[2] - box[0]) * w,
                                             (box[3] - box[1]) * h, linewidth=1, edgecolor='r', facecolor='none')
                    listRect.append(rect)
                    # Add the patch to the Axes
                    ax.add_patch(rect)

                plt.show()
                plt.axis('off')
                plt.pause(0.1)
                for rec in listRect:
                    rec.remove()
                id += 1

def Draw(imsPath, bboxs):
    fig, ax = plt.subplots(1)
    ab = None

    for imPath, bbox in zip(imsPath, bboxs):
        rawIm = cv2.imread(imPath)

        # raw = numpy.zeros((1920, 1920, 3), dtype = 'uint8')
        # raw[0:1080, :, :] = rawIm
        # rawIm = raw

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
    CreateLSTMModel()
    CreateSSDExtractFactory()
    TestModel()