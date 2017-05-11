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
DISPLAY_FREQUENCY = 10
SAVE_FREQUENCY    = 150

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = 10
NUM_HIDDEN        = 512
INPUTS_SIZE       = [256]
OUTPUTS_SIZE      = [6, 24]
SEQUENCE_TRAIN    = NUM_TRUNCATE * 2

# BOUNDING BOX HYPER
ALPHA = 0.6
BETA  = 1.4

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# SAVE MODEL PATH
SAVE_PATH       = '../Pretrained/SSD512/LSTM_SSD_Epoch=%d_Iter=%d.pkl'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/SSD512/LSTM_SSD_Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 1
START_ITERATION = 7050

#  GLOBAL VARIABLES
Dataset           = None
FeatureFactory    = None
DefaultBboxs      = None
BoxsVariances     = None
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
    FeatureFactory = SSD512FeaExtraction(batchSize = NUM_TRUNCATE)
    FeatureFactory.LoadCaffeModel('../Models/SSD_512x512/VOC0712/deploy.prototxt',
                                  '../Models/SSD_512x512/VOC0712/VGG_coco_SSD_512x512_iter_360000.caffemodel')
    FeatureFactory.LoadEncodeLayers('../Preprocessing/SSD512/ssd512_conv4_3_norm_encode.pkl',
                                    '../Preprocessing/SSD512/ssd512_fc7_encode.pkl',
                                    '../Preprocessing/SSD512/ssd512_conv6_2_encode.pkl')
    DefaultBboxs = FeatureFactory.GetDefaultBbox(imageWidth = 512,
                                                 sMin       = 10,
                                                 sMax       = 90,
                                                 layerSizes = [(64, 64),
                                                               (32, 32),
                                                               (16, 16),
                                                               ( 8,  8),
                                                               ( 4,  4),
                                                               ( 2,  2),
                                                               ( 1,  1)],
                                                 numBoxs    = [6, 6, 6, 6, 6, 6, 6],
                                                 offset     = 0.5,
                                                 steps      = [8, 16, 32, 64, 128, 256, 512])
    BoxsVariances = numpy.zeros((DefaultBboxs.__len__() * DefaultBboxs[0].__len__(), 4), dtype = 'float32')
    BoxsVariances[:, 0] = 0.1;      BoxsVariances[:, 1] = 0.1;
    BoxsVariances[:, 2] = 0.2;      BoxsVariances[:, 3] = 0.2;


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
            minInterestBox, maxInterestBox = InterestBox2(archorBox, groundTruth, ALPHA, BETA)
            if minInterestBox >= 0.5 and maxInterestBox >= 0.1:
            # if InterestBox1(archorBox, groundTruth, ALPHA, BETA):
                pred[bboxIdx][archorboxIdx] = 1
                gt[bboxIdx][archorboxIdx]   = [(groundTruth[0] - archorBox[0]) / archorBox[2],
                                               (groundTruth[1] - archorBox[1]) / archorBox[3],
                                                math.log(groundTruth[2] / archorBox[2]),
                                                math.log(groundTruth[3] / archorBox[3])]
    return [pred, gt]

def CheckPred(defaultBboxs, groundTruth, overlapThres):
    for bboxIdx, dfbbox in enumerate(defaultBboxs):
        for archorboxIdx, archorBox in enumerate(dfbbox):
            minInterestBox, maxInterestBox = InterestBox2(archorBox, groundTruth, ALPHA, BETA)
            if minInterestBox >= 0.5 and maxInterestBox >= 0.1:
            # if InterestBox1(archorBox, groundTruth, ALPHA, BETA):
                return True
    return False


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


def FilterBboxs(imsPath,
                bboxs,
                defaultBboxs = None,
                overlapThres = 0.5):
    newImsPath = []
    newBboxs   = []
    count      = 0
    for (imPath, bbox) in zip(imsPath, bboxs):
        count += 1
        if CheckPred(defaultBboxs, bbox, overlapThres) == True:
            newImsPath.append(imPath)
            newBboxs.append(bbox)
        print "\r    Filter metadata: %d / %d. Filted metadata: %d sample(s)" % (count, bboxs.__len__(), newBboxs.__len__()),
    print "Filter completed !"
    return newImsPath, newBboxs

########################################################################################################################
#                                                                                                                      #
#    TRAIN LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def TrainModel():
    global Dataset, LSTMModel, FeatureFactory, DefaultBboxs, BoxsVariances

    # Create startStateS | startStateC
    startStateS = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    startStateC = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')

    # Plot training cost
    iterVisualize = []
    costVisualize = []
    plt.ion()
    data, = plt.plot(iterVisualize, costVisualize)
    plt.axis([START_ITERATION, START_ITERATION + 10, 0, 10])

    # Load model
    if CheckFileExist(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION),
                      throwError = False) == True:
        file = open(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION))
        LSTMModel.LoadModel(file)
        file.close()
        print ('Load model !')

    RatioPosNeg = 2.

    # Train each folder in train folder
    iter = START_ITERATION
    costs = []
    predictPostAves = []
    predictLocAves  = []
    predictNegAves  = []

    Dataset.DataOpts['data_phase'] = 'train'
    allFolderNames = Dataset.GetAllFolderNames()
    for epoch in xrange(0, NUM_EPOCH):
        if epoch < START_EPOCH:
            continue
        for folderIdx, folderName in enumerate(allFolderNames):
            Dataset.DataOpts['data_folder_name'] = folderName
            Dataset.DataOpts['data_folder_type'] = 'gt'
            allObjectIds = Dataset.GetAllObjectIds()

            for objectId in allObjectIds:
                Dataset.DataOpts['data_object_id'] = objectId

                print ('Load metadata of objectId = %d' % (objectId))
                imsPath, bboxs = Dataset.GetSequenceBy(occluderThres = 0.5)

                # If number image in sequence less than NUM_TRUNCATE => we choose another sequence to train
                if (imsPath.__len__() < NUM_TRUNCATE):
                    continue

                # startIdx = numpy.random.randint(imsPath.__len__() - SEQUENCE_TRAIN + 1)
                # endIdx   = startIdx + SEQUENCE_TRAIN
                # imsPath  = imsPath[startIdx: endIdx]
                # bboxs    = bboxs[startIdx: endIdx]

                imsPath, bboxs = FilterBboxs(imsPath,
                                             bboxs,
                                             defaultBboxs = DefaultBboxs,
                                             overlapThres = 0.5)

                # Else train the sequence ...................
                S = startStateS;    C = startStateC
                NUM_BATCH = (imsPath.__len__()) // NUM_TRUNCATE
                for batchId in range(NUM_BATCH):
                    # Get batch
                    imsPathBatch = imsPath[batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE]
                    bboxsBatch   = bboxs[batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE]

                    # Extract feature and prepare bounding box before training....
                    batchFeatures          = FeatureFactory.ExtractFeature(imsPath = imsPathBatch)   # Extract batch features
                    batchPreds, batchBboxs = CompareBboxs(DefaultBboxs, bboxsBatch)

                    inputBatchFeatures   = batchFeatures[0 : NUM_TRUNCATE]
                    outputPreds          = batchPreds[0 : NUM_TRUNCATE]
                    outputBboxs          = batchBboxs[0 : NUM_TRUNCATE]

                    numFeaturesPerIm   = batchFeatures.shape[1]
                    numAnchorBoxPerLoc = DefaultBboxs.shape[1]

                    inputBatchFeatureGts = GetFeatures(inputBatchFeatures, outputPreds)
                    outputPreds = outputPreds.reshape((NUM_TRUNCATE, numFeaturesPerIm, numAnchorBoxPerLoc * 1))
                    outputBboxs = outputBboxs.reshape((NUM_TRUNCATE, numFeaturesPerIm, numAnchorBoxPerLoc * 4))

                    print ('Load batch ! Done !')

                    for k in range(1):
                        inputBatchFeatureGt = GetRandomFeatures(inputBatchFeatureGts)

                        iter += 1
                        cost, newS, newC, predictPostAve, predictLocAve, predictNegAve, k0, k1, k2, k3, k4 = LSTMModel.TrainFunc(inputBatchFeatureGt,
                                                               inputBatchFeatures,
                                                               outputPreds,
                                                               outputBboxs,
                                                               S,
                                                               C,
                                                               BoxsVariances,
                                                               RatioPosNeg)
                        costs.append(cost)
                        predictPostAves.append(predictPostAve)
                        predictLocAves.append(predictLocAve)
                        predictNegAves.append(predictNegAve)

                        # Check ratioPosNeg
                        # if iter % DISPLAY_FREQUENCY == 0:
                        #     if numpy.mean(predictPostAves) < 0.5:
                        #         RatioPosNeg = 1. / 3
                        #     else:
                        #         RatioPosNeg = 3

                        if iter % DISPLAY_FREQUENCY == 0:
                            print ('Epoch = %d, iteration = %d, cost = %f, predictPosAve = %f, predictLocAve = %f, predictNegAve = %f. ObjectId = %d' % (epoch, iter, numpy.mean(costs),
                                                                                                                                     numpy.mean(predictPostAves),
                                                                                                                                     numpy.mean(predictLocAves),
                                                                                                                                     numpy.mean(predictNegAves), objectId))
                            iterVisualize.append(iter)
                            costVisualize.append(numpy.mean(costs))

                            data.set_xdata(numpy.append(data.get_xdata(), iterVisualize[-1]))
                            data.set_ydata(numpy.append(data.get_ydata(), costVisualize[-1]))
                            yLimit = math.floor(numpy.max(costVisualize) / 10) * 10 + 4
                            plt.axis([START_ITERATION, iterVisualize[-1], 0, yLimit])
                            plt.draw()
                            plt.pause(0.05)
                            costs = []
                            predictPostAves = []
                            predictLocAves  = []
                            predictNegAves  = []

                        if iter % SAVE_FREQUENCY == 0:
                            file = open(SAVE_PATH % (epoch, iter), 'wb')
                            LSTMModel.SaveModel(file)
                            file.close()
                            print ('Save model !')

                    S = newS;       C = newC


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

        plt.pause(0.05)

        rect.remove()

if __name__ == '__main__':
    LoadDataset()
    CreateLSTMModel()
    CreateSSDExtractFactory()
    TrainModel()