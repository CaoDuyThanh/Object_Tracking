import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from FeaturesExtraction.HOGExtraction import *
from Utils.MOTDataHelper import *
from Models.LSTM.LSTMTrackingModel import *
from Utils.DefaultBox import *
from Utils.BBoxHelper import *

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = 1
NUM_HIDDEN        = 512
INPUTS_SIZE       = [576]
OUTPUTS_SIZE      = [1, 24]

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'

# SAVE FILE
SAVE_PATH = '../Pretrained/LSTM_HOG_Epoch=%d_Iter=%d.pkl'

#  GLOBAL VARIABLES
Dataset           = None
HOGExtractFactory = None
LSTMModel         = None

def LoadDataset():
    global Dataset
    Dataset = MOTDataHelper(DATASET_PATH)

def CreateHOGExtractFactory():
    global HOGExtractFactory
    HOGExtractFactory = HOGExtraction()

def CreateLSTMModel():
    global LSTMModel
    LSTMModel = LSTMTrackingModel(numTruncate = NUM_TRUNCATE,
                                  numHidden   = NUM_HIDDEN,
                                  inputsSize  = INPUTS_SIZE,
                                  outputsSize = OUTPUTS_SIZE)

def createGroundTruth(defaultBbox, bbox):
    pred = numpy.zeros(defaultBbox.shape[0], dtype = 'float32')
    gt   = numpy.zeros(defaultBbox.shape, dtype = 'float32')
    for bboxIdx, dfbbox in enumerate(defaultBbox):
        prediction = 0
        for archorboxIdx, archorBox in enumerate(dfbbox):
            iou = IOU(archorBox, bbox)
            if iou >= 0.5:
                prediction = 1
                gt[bboxIdx][archorboxIdx] = [(bbox[0] - archorBox[0]) / archorBox[2],
                                             (bbox[1] - archorBox[1]) / archorBox[3],
                                             math.log(bbox[2] / archorBox[2]),
                                             math.log(bbox[3] / archorBox[3])]
            else:
                gt[bboxIdx][archorboxIdx] = [0, 0, 0, 0]

        pred[bboxIdx] = prediction

    return [pred, gt]


def CreateTestData(imsHOGFeatures, defaultBboxs, bboxs):
    features = []
    preds = []
    gts = []
    for idx, imHOGFeatures in enumerate(imsHOGFeatures):
        bbox = bboxs[idx]

        feature = []
        for imHOGFeature in imHOGFeatures:
            feature.append(numpy.asarray(imHOGFeature[0], dtype='float32'))
        feature = numpy.concatenate(feature, axis=0)
        pred, gt = createGroundTruth(defaultBboxs, bbox)

        features.append(feature)
        preds.append(pred)
        gts.append(gt)

    features = numpy.asarray(features, dtype='float32')
    preds = numpy.asarray(preds, dtype='float32')
    gts = numpy.asarray(gts, dtype='float32')

    return features, preds, gts

def CreateTest1(imsHOGFeatures):
    features = []
    for idx, imHOGFeatures in enumerate(imsHOGFeatures):
        feature = []
        for imHOGFeature in imHOGFeatures:
            feature.append(numpy.asarray(imHOGFeature[0], dtype='float32'))
        feature = numpy.concatenate(feature, axis=0)
        features.append(feature)

    features = numpy.asarray(features, dtype='float32')
    return features


def getFeatures(features, preds):
    featuresgts = []
    for (feature, pred) in zip(features, preds):
        ftgt = []
        for idx in range(feature.shape[0]):
            p = pred[idx]

            if p == 1:
                ftgt.append(feature[idx])
        featuresgts.append(ftgt)

    return featuresgts


def TestModel():
    global Dataset, LSTMModel, HOGExtractFactory

    # Create startStateS | startStateC
    startStateS = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    startStateC = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')

    # Load model
    file = open(SAVE_PATH % (1, 32000))
    LSTMModel.LoadModel(file)
    file.close()
    print ('Load model !')

    # Create default boxs
    defaultBboxs   = None

    # Test each Folder
    Dataset.DataOpts['data_phase'] = 'test'
    allFolderNames = Dataset.GetAllFolderNames()
    for folderName in allFolderNames:
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_folder_type'] = 'det'
        imsPath, bboxs = Dataset.GetRandomBbox()

        S = startStateS
        C = startStateC
        while True:
            imsHOGFeatures, defaultBboxs = HOGExtractFactory.ExtractFeature(imPaths      = [imsPath[1]],
                                                                            defaultBboxs = defaultBboxs)

            features, preds, gts = CreateTestData(imsHOGFeatures, defaultBboxs, [bboxs])
            featuresgts = getFeatures(features, preds)
            featuresgt = getRandomFeatures(featuresgts)
            S, C = LSTMModel.NextState(featuresgt, S, C)
            id = 2
            while id < imsPath.__len__():
                imPath = imsPath[id]
                imsHOGFeatures, defaultBboxs = HOGExtractFactory.ExtractFeature(imPaths      = [imPath],
                                                                                defaultBboxs = defaultBboxs)
                features = CreateTest1(imsHOGFeatures)
                predConfs, predBboxs = LSTMModel.PredFunc(features, S, C)
                predBboxs = predBboxs.reshape((predBboxs.shape[0], 6, 4))
                bestBboxs = getBboxs(defaultBboxs, predConfs, predBboxs, 0.3)

                rawIm = cv2.imread(imPath)
                _, w, h = rawIm.shape
                fig, ax = plt.subplots(1)
                ax.imshow(rawIm)

                for box in bestBboxs:
                    rect = patches.Rectangle((box[0] * w, box[1] * h), (box[2] - box[0]) * w,
                                             (box[3] - box[1]) * h, linewidth=1, edgecolor='r', facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)

                plt.show()
                plt.axis('off')


def getRandomFeatures(features):
    featuresgt = numpy.zeros((features.__len__(), features[0][0].shape[0]), dtype='float32')
    for idx in range(features.__len__()):
        featuresgt[idx] = features[idx][numpy.random.randint(features[idx].__len__())]

    return featuresgt


def getBboxs(defaultBboxs, predConfs, predBboxs, threshold):
    bestBoxes = []
    for idx, defaultBbox in enumerate(defaultBboxs):
        if predConfs[idx] >= threshold:
            boxs = predBboxs[idx]
            for archorBox, box in zip(defaultBbox, boxs):
                cx = archorBox[0]
                cy = archorBox[1]
                w  = archorBox[2]
                h  = archorBox[3]

                offsetXmin = box[0]
                offsetYmin = box[1]
                offsetXmax = box[2]
                offsetYmax = box[3]

                cx = offsetXmin * w + cx
                cy = offsetYmin * h + cy
                w = math.exp(offsetXmax) * w
                h = math.exp(offsetYmax) * h

                xmin = cx - w / 2.
                ymin = cy - h / 2.
                xmax = cx + w / 2.
                ymax = cy + h / 2.

                xmin = min(max(xmin, 0), 1)
                ymin = min(max(ymin, 0), 1)
                xmax = min(max(xmax, 0), 1)
                ymax = min(max(ymax, 0), 1)

                bestBoxes.append([xmin, ymin, xmax, ymax])
    return bestBoxes

if __name__ == '__main__':
    LoadDataset()
    CreateLSTMModel()
    CreateHOGExtractFactory()
    TestModel()