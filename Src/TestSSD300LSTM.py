import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from FeaturesExtraction.SSD300FeaExtraction import *
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
BATCH_SIZE        = 1
DISPLAY_FREQUENCY = 50

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = (1, 1)
NUM_HIDDEN        = 2048
INPUTS_SIZE       = [256 + 4 + 256 + 4 + 256 + 4 + 256 + 4 + 256 + 4 + 256 + 4]
OUTPUTS_SIZE      = [1, 4]

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/SSD/LSTM_SSD_Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 14
START_ITERATION = 265000

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
    global FeatureFactory, DefaultBboxs, BoxsVariances
    FeatureFactory = SSD300FeaExtraction()
    FeatureFactory.LoadCaffeModel('../Models/SSD_300x300/VOC0712/deploy.prototxt',
                                  '../Models/SSD_300x300/VOC0712/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    DefaultBboxs = FeatureFactory.GetDefaultBbox(imageWidth = 300,
                                                 sMin       = 10,
                                                 sMax       = 100,
                                                 layerSizes = [(38, 38),
                                                               (19, 19),
                                                               (10, 10),
                                                               (5, 5),
                                                               (3, 3),
                                                               (1, 1)],
                                                 numBoxs    = [4, 4, 4, 4, 4, 4],
                                                 offset     = 0.5,
                                                 steps      = [8, 16, 32, 64, 100, 300])
    BoxsVariances = numpy.zeros((DefaultBboxs.__len__(), 4), dtype='float32')
    BoxsVariances[:, 0] = 0.1;
    BoxsVariances[:, 1] = 0.1;
    BoxsVariances[:, 2] = 0.2;
    BoxsVariances[:, 3] = 0.2;


########################################################################################################################
#                                                                                                                      #
#    CREATE LSTM MODEL FOR TRACKING OBJECTS                                                                            #
#                                                                                                                      #
########################################################################################################################
def CreateLSTMModel():
    global LSTMModel, FeatureFactory
    LSTMModel = LSTMTrackingModel(batchSize   = BATCH_SIZE,
                                  numTruncate = NUM_TRUNCATE,
                                  numHidden   = NUM_HIDDEN,
                                  inputsSize  = INPUTS_SIZE,
                                  outputsSize = OUTPUTS_SIZE,
                                  featureFactory = FeatureFactory,
                                  featureXBatch  = FeatureFactory.Net.Layer['features_reshape'].Output.reshape((BATCH_SIZE, numpy.sum(NUM_TRUNCATE), 1940, 256)))

########################################################################################################################
#                                                                                                                      #
#    UTILITIES (MANY DIRTY CODES)                                                                                      #
#                                                                                                                      #
########################################################################################################################
def GetBatchMetaData(batchObjectIds):
    global Dataset

    imsPathBatch = []
    bboxsBatch   = []
    maxSequence  = 0
    for oneObjectId in batchObjectIds:
        folderName = oneObjectId[0]
        objectId   = int(oneObjectId[1])
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_object_id']   = objectId
        imsPath, bboxs = Dataset.GetSequenceBy(occluderThres=0.5)
        imsPathBatch.append(imsPath)
        bboxsBatch.append(bboxs)
        maxSequence = max(maxSequence, imsPath.__len__())
    for (imsPath, bboxs) in zip(imsPathBatch, bboxsBatch):
        if imsPath.__len__() < maxSequence:
            for id in range(maxSequence - imsPath.__len__()):
                imsPath.append('')
                bboxs.append([0, 0, 0, 0])

    return imsPathBatch, bboxsBatch, maxSequence


def GetBatchData(imsPathBatch, bboxsBatch):
    global FeatureFactory, DefaultBboxs

    inputXBatch = numpy.zeros((BATCH_SIZE, numpy.sum(NUM_TRUNCATE), 3, 300, 300), dtype='float32')
    YsBatch     = numpy.zeros((BATCH_SIZE, numpy.sum(NUM_TRUNCATE), 1940, 1), dtype='float32')
    bboxYsBatch = numpy.zeros((BATCH_SIZE, numpy.sum(NUM_TRUNCATE), 1940, 4), dtype='float32')

    for batchId in range(imsPathBatch.__len__()):
        imsPathSequence = imsPathBatch[batchId]
        inputSequence   = ReadImages(imsPath = imsPathSequence, batchSize = numpy.sum(NUM_TRUNCATE))

        bboxsSequence              = bboxsBatch[batchId]
        YsSequence, bboxYsSequence = CreateBboxYs(DefaultBboxs, bboxsSequence)

        inputXBatch[batchId, :, :, :] = inputSequence
        YsBatch[batchId, :, :, :]     = YsSequence
        bboxYsBatch[batchId, :, :, :] = bboxYsSequence
    inputXBatch = inputXBatch.reshape((BATCH_SIZE * numpy.sum(NUM_TRUNCATE), 3, 300, 300))

    return inputXBatch, YsBatch, bboxYsBatch


def CreateFeatureEncodeX(feature, Ys, bboxYs):
    return 0

def CreateBboxYs(defaultBboxs, bboxs):
    Ys     = []
    bboxYs = []

    for bbox in bboxs:
        Y, bboxY = CreateGroundTruth(defaultBboxs, bbox)
        Ys.append(Y)
        bboxYs.append(bboxY)

    Ys     = numpy.asarray(Ys, dtype = 'float32')
    bboxYs = numpy.asarray(bboxYs, dtype = 'float32')

    return Ys, bboxYs

def CreateGroundTruth(defaultBboxs, groundTruth):
    global BoxsVariances

    Y     = numpy.zeros((defaultBboxs.shape[0], 1), dtype = 'float32')
    bboxY = numpy.zeros((defaultBboxs.shape[0], 4), dtype='float32')

    if (groundTruth[2] < 0.00001 and groundTruth[3] < 0.00001):
        return Y, bboxY

    inX   = (defaultBboxs[:, 0] - defaultBboxs[:, 2] / 2 < groundTruth[0]) * \
            (defaultBboxs[:, 0] + defaultBboxs[:, 2] / 2 > groundTruth[0])
    inY   = (defaultBboxs[:, 1] - defaultBboxs[:, 3] / 2 < groundTruth[1]) * \
            (defaultBboxs[:, 1] + defaultBboxs[:, 3] / 2 > groundTruth[1])
    inXY  = inX * inY
    Y[inXY] = 1

    bboxY[:, 0] = (groundTruth[0] - defaultBboxs[:, 0]) / defaultBboxs[:, 2] / BoxsVariances[:, 0]
    bboxY[:, 1] = (groundTruth[1] - defaultBboxs[:, 1]) / defaultBboxs[:, 3] / BoxsVariances[:, 1]
    bboxY[:, 2] = numpy.log(groundTruth[2] / defaultBboxs[:, 2]) / BoxsVariances[:, 2]
    bboxY[:, 3] = numpy.log(groundTruth[3] / defaultBboxs[:, 3]) / BoxsVariances[:, 3]
    bboxY = bboxY * Y

    return Y, bboxY

def SortData(data):
    global Dataset
    data = numpy.array(data)

    dataSizes = numpy.zeros((data.__len__(),))
    for idx, sample in enumerate(data):
        folderName = sample[0]
        objectId   = int(sample[1])

        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_object_id']   = objectId
        imsPath, _ = Dataset.GetSequenceBy(occluderThres=0.5)
        dataSizes[idx] = imsPath.__len__()

    sortedIdx = numpy.argsort(dataSizes, axis = 0)
    data = data[sortedIdx]

    return data

def getBboxs(defaultBboxs, predConfs, predBboxs, threshold):
    bestBoxes = []
    bestConf  = 0.0
    bestBbox  = None
    deBbox    = []
    for idx, predConf in enumerate(predConfs):
        if idx >= 1930:
            break
        if predConf >= threshold:
            archorBox = defaultBboxs[idx]
            box       = predBboxs[idx]

            cx = archorBox[0]
            cy = archorBox[1]
            w  = archorBox[2]
            h  = archorBox[3]

            offsetXmin = box[0]
            offsetYmin = box[1]
            offsetXmax = box[2]
            offsetYmax = box[3]

            cx1 = offsetXmin * 0.1 * w + cx
            cy1 = offsetYmin * 0.1 * h + cy
            w1  = math.exp(offsetXmax * 0.2) * w
            h1  = math.exp(offsetYmax * 0.2) * h

            # xmin = cx1 - w1 / 2.
            # ymin = cy1 - h1 / 2.
            # xmax = cx1 + w1 / 2.
            # ymax = cy1 + h1 / 2.
            #
            # xmin = min(max(xmin, 0), 1)
            # ymin = min(max(ymin, 0), 1)
            # xmax = min(max(xmax, 0), 1)
            # ymax = min(max(ymax, 0), 1)

            # bestBoxes.append([xmin, ymin, xmax, ymax])
            bestBoxes.append([cx1, cy1, w1, h1])
            deBbox.append([cx, cy, w, h])

            if bestConf < predConf:
                bestConf = predConf
                # bestBbox = [xmin, ymin, xmax, ymax]
                bestBbox = [cx1, cy1, w1, h1]

    return bestBbox, bestBoxes, deBbox

########################################################################################################################
#                                                                                                                      #
#    TEST LSTM MODEL........................                                                                           #
#                                                                                                                      #
########################################################################################################################
def TestModel():
    global Dataset, LSTMModel, FeatureFactory, DefaultBboxs

    # Create startStateS | startStateC
    startStateC = numpy.zeros((BATCH_SIZE, LSTMModel.EncodeNet.LayerOpts['lstm_num_hidden'],), dtype='float32')
    startStateH = numpy.zeros((BATCH_SIZE, LSTMModel.EncodeNet.LayerOpts['lstm_num_hidden'],), dtype='float32')

    # Plot training cost
    plt.ion()
    fig, ax = plt.subplots(1)
    ab = None

    # Load Model
    if CheckFileExist(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION),
                      throwError=False) == True:
        file = open(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION))
        LSTMModel.LoadModel(file)
        file.close()
        print ('Load model !')

    # Test each Folder
    Dataset.DataOpts['data_phase'] = 'train'
    allFolderNames = Dataset.GetAllFolderNames()
    for folderName in allFolderNames:
        folderName = 'MOT16-04'
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_folder_type'] = 'det'
        imsPath, bboxs = Dataset.GetRandomBbox()

        # Reset status at the beginning of each sequence
        encodeC = startStateC;
        encodeH = startStateH;
        decodeC = startStateC;

        imPath = imsPath[1]
        rawIm = cv2.imread(imPath)
        h, w, _ = rawIm.shape

        if ab == None:
            ab = ax.imshow(rawIm)
        else:
            ab.set_data(rawIm)

        listRect = []
        box = bboxs
        rect = patches.Rectangle(((box[0] - box[2] / 2) * w, (box[1] - box[3] / 2) * h),
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

        id = 1
        while id < imsPath.__len__():
            # Extract feature and prepare bounding box before training....
            FeatureEncodeXBatch, YsBatch, BboxYsBatch = GetBatchData([[imsPath[id], imsPath[id+1]]], [[bboxs]])

            # Draw([[imsPath[id], imsPath[id+1]]], YsBatch, BboxYsBatch)

            result = LSTMModel.PredFunc(FeatureEncodeXBatch,
                                        YsBatch,
                                        BboxYsBatch,
                                        encodeC,
                                        encodeH,
                                        decodeC)
            predConfs  = result[0]
            predBboxs  = result[1]
            newEncodeC = result[2 : 2 + BATCH_SIZE]
            newEncodeH = result[2 + BATCH_SIZE : 2 + 2 * BATCH_SIZE]

            predBboxs = predBboxs.reshape((1940, 4))
            bestBbox, bestBboxs, debbox = getBboxs(DefaultBboxs, predConfs, predBboxs, 0.5)

            imPath = imsPath[id]
            rawIm = cv2.imread(imPath)
            h, w, _ = rawIm.shape

            if ab == None:
                ab = ax.imshow(rawIm)
            else:
                ab.set_data(rawIm)

            listRect = []
            for box in bestBboxs:
                box = bestBbox
                rect = patches.Rectangle(((box[0] - box[2] / 2) * w, (box[1] - box[3] / 2) * h),
                                           box[2] * w,
                                           box[3] * h, linewidth=1, edgecolor='r', facecolor='none')
                listRect.append(rect)
                # Add the patch to the Axes
                ax.add_patch(rect)
                break

            plt.show()
            plt.axis('off')
            plt.pause(0.1)
            for rec in listRect:
                rec.remove()
            id += 1

            bboxs = bestBbox

            encodeC = numpy.asarray(newEncodeC, dtype='float32');
            encodeH = numpy.asarray(newEncodeH, dtype='float32');


def Draw(imsPathMiniBatch, YsBatch, BboxYsBatch):
    global DefaultBboxs

    fig, ax = plt.subplots(1)
    ab = None

    imsPathMini = imsPathMiniBatch[0]
    Ys = YsBatch[0]
    BboxYs = BboxYsBatch[0]

    for i in range(Ys.shape[0]):
        Y = Ys[i]
        BboxY = BboxYs[i]

        rawIm = cv2.imread(imsPathMini[i])

        # raw = numpy.zeros((1920, 1920, 3), dtype = 'uint8')
        # raw[0:1080, :, :] = rawIm
        # rawIm = raw

        if ab == None:
            ab = ax.imshow(rawIm)
        else:
            ab.set_data(rawIm)

        for idx, y in enumerate(Y):
            if (y == 1):
                cx1 = DefaultBboxs[idx][0]
                cy1 = DefaultBboxs[idx][1]
                w1 = DefaultBboxs[idx][2]
                h1 = DefaultBboxs[idx][3]

                cx = BboxY[idx][0]
                cy = BboxY[idx][1]
                w = BboxY[idx][2]
                h = BboxY[idx][3]

                cx2 = cx * 0.1 * w1 + cx1
                cy2 = cy * 0.1 * h1 + cy1
                w2 = numpy.exp(w * 0.2) * w1
                h2 = numpy.exp(h * 0.2) * h1
                box0 = [cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2]
                box = [cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2]

                h, w, _ = rawIm.shape
                rect0 = patches.Rectangle((box0[0] * w, box0[1] * h), (box0[2] - box0[0]) * w,
                                          (box0[3] - box0[1]) * h, linewidth=1, edgecolor='r', facecolor='none')
                rect = patches.Rectangle((box[0] * w, box[1] * h), (box[2] - box[0]) * w,
                                         (box[3] - box[1]) * h, linewidth=1, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect0)
                ax.add_patch(rect)

                plt.show()
                plt.axis('off')

                plt.pause(0.05)
                rect0.remove()
                rect.remove()


if __name__ == '__main__':
    LoadDataset()
    CreateSSDExtractFactory()
    CreateLSTMModel()
    TestModel()