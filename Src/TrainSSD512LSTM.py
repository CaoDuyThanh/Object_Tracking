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
NUM_EPOCH         = 10
DISPLAY_FREQUENCY = 50
SAVE_FREQUENCY    = 1000

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = 5
NUM_HIDDEN        = 512
INPUTS_SIZE       = [256]
OUTPUTS_SIZE      = [6, 24]

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# SAVE MODEL PATH
SAVE_PATH       = '../Pretrained/SSD/LSTM_SSD_Epoch=%d_Iter=%d.pkl'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/SSD/LSTM_SSD_Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 6
START_ITERATION = 52000

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
    FeatureFactory = SSD300FeaExtraction(batchSize =NUM_TRUNCATE + 1)
    FeatureFactory.LoadCaffeModel('../Models/SSD_300x300/VOC0712/deploy.prototxt',
                                  '../Models/SSD_300x300/VOC0712/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    FeatureFactory.LoadEncodeLayers('../Preprocessing/conv4_3_norm_encode.pkl',
                                    '../Preprocessing/fc7_encode.pkl',
                                    '../Preprocessing/conv6_2_encode.pkl')
    DefaultBboxs = FeatureFactory.GetDefaultBbox(imageWidth = 300,
                                                 sMin       = 3,
                                                 sMax       = 90,
                                                 layerSizes = [(38, 38),
                                                               (19, 19),
                                                               (10, 10),
                                                               (5, 5),
                                                               (3, 3),
                                                               (1, 1)],
                                                 numBoxs    = [6, 6, 6, 6, 6, 6],
                                                 offset     = 0.5,
                                                 steps      = [8, 16, 32, 64, 100, 300])


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
            iou = IOU(archorBox, groundTruth)
            if iou >= 0.5:
                pred[bboxIdx][archorboxIdx] = 1
                gt[bboxIdx][archorboxIdx]   = [(groundTruth[0] - archorBox[0]) / archorBox[2],
                                               (groundTruth[1] - archorBox[1]) / archorBox[3],
                                                math.log(groundTruth[2] / archorBox[2]),
                                                math.log(groundTruth[3] / archorBox[3])]
    return [pred, gt]


def GetFeatures(features, preds):
    featuresgts = []
    for (feature, pred) in zip(features, preds):
        ftgt = []
        for idx in range(feature.shape[0]):
            p = numpy.max(pred[idx])

            if p == 1:
                ftgt.append(feature[idx])
        featuresgts.append(ftgt)
    return featuresgts


def GetRandomFeatures(features):
    featuresgt = numpy.zeros((features.__len__(), features[0][0].shape[0]), dtype='float32')
    for idx in range(features.__len__()):
        featuresgt[idx] = features[idx][numpy.random.randint(features[idx].__len__())]

    return featuresgt



########################################################################################################################
#                                                                                                                      #
#    TRAIN LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def TrainModel():
    global Dataset, LSTMModel, FeatureFactory, DefaultBboxs

    # Create startStateS | startStateC
    startStateS = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    startStateC = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    # startStateS = None
    # startStateC = None

    # Plot training cost
    iterVisualize = []
    costVisualize = []
    plt.ion()
    data, = plt.plot(iterVisualize, costVisualize)
    plt.axis([START_ITERATION, START_ITERATION + 10, 0, 10])

    # Load model
    file = open(SAVE_PATH % (START_EPOCH, START_ITERATION))
    LSTMModel.LoadModel(file)
    file.close()
    print ('Load model !')

    # Train each folder in train folder
    defaultBboxs = None
    iter = START_ITERATION
    costs = []

    Dataset.DataOpts['data_phase'] = 'train'
    allFolderNames = Dataset.GetAllFolderNames()
    for folderName in allFolderNames:
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_folder_type'] = 'gt'
        allObjectIds = Dataset.GetAllObjectIds()

        for epoch in range(NUM_EPOCH):
            if epoch < START_EPOCH:
                continue
            for objectId in allObjectIds:
                Dataset.DataOpts['data_object_id'] = objectId
                imsPath, bboxs = Dataset.GetSequenceBy()

                # If number image in sequence less than NUM_TRUNCATE => we choose another sequence to train
                if (imsPath.__len__() < NUM_TRUNCATE):
                    continue

                # Else we train...................
                S = startStateS;    C = startStateC
                NUM_BATCH = (imsPath.__len__() - 1) // NUM_TRUNCATE
                for batchId in range(NUM_BATCH):
                    # Get batch
                    imsPathBatch = imsPath[batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE + 1]
                    bboxsBatch   = bboxs[batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE + 1]

                    # Extract feature and prepare bounding box before training....
                    batchFeatures          = FeatureFactory.ExtractFeature(imsPath = imsPathBatch)   # Extract batch features
                    batchPreds, batchBboxs = CompareBboxs(DefaultBboxs, bboxsBatch)

                    inputBatchFeatures   = batchFeatures[0 : NUM_TRUNCATE]
                    outputPreds          = batchPreds[1 : ]
                    outputBboxs          = batchBboxs[1 : ]

                    numFeaturesPerIm   = batchFeatures.shape[1]
                    numAnchorBoxPerLoc = DefaultBboxs.shape[1]

                    inputBatchFeatureGts = GetFeatures(inputBatchFeatures, outputPreds)
                    outputPreds = outputPreds.reshape((NUM_TRUNCATE, numFeaturesPerIm, numAnchorBoxPerLoc * 1))
                    outputBboxs = outputBboxs.reshape((NUM_TRUNCATE, numFeaturesPerIm, numAnchorBoxPerLoc * 4))

                    print ('Load batch ! Done !')

                    test = True
                    for ft in inputBatchFeatureGts:
                        if ft.__len__() == 0:
                            test = False
                            break
                    if test == False:
                        continue

                    for k in range(10):
                        inputBatchFeatureGt = GetRandomFeatures(inputBatchFeatureGts)

                        iter += 1
                        cost, newS, newC = LSTMModel.TrainFunc(inputBatchFeatureGt,
                                                               inputBatchFeatures,
                                                               outputPreds,
                                                               outputBboxs,
                                                               S,
                                                               C)
                        costs.append(cost)

                        if iter % DISPLAY_FREQUENCY == 0:
                            print ('Epoch = %d, iteration = %d, cost = %f. ObjectId = %d' % (epoch, iter, numpy.mean(costs), objectId))
                            iterVisualize.append(iter)
                            costVisualize.append(numpy.mean(costs))

                            data.set_xdata(numpy.append(data.get_xdata(), iterVisualize[-1]))
                            data.set_ydata(numpy.append(data.get_ydata(), costVisualize[-1]))
                            plt.axis([START_ITERATION, iterVisualize[-1], 0, 10])
                            plt.draw()
                            plt.pause(0.05)
                            costs = []

                        if iter % SAVE_FREQUENCY == 0:
                            file = open(SAVE_PATH % (epoch, iter), 'wb')
                            LSTMModel.SaveModel(file)
                            file.close()
                            print ('Save model !')

                    S = newS;       C = newC


if __name__ == '__main__':
    LoadDataset()
    CreateLSTMModel()
    CreateSSDExtractFactory()
    TrainModel()