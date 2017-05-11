import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from FeaturesExtraction.HOGExtraction import *
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
INPUTS_SIZE       = [576]
OUTPUTS_SIZE      = [1, 24]

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'

# SAVE MODEL PATH
SAVE_PATH = '../Pretrained/LSTM_HOG_Epoch=%d_Iter=%d.pkl'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/LSTM_HOG_Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 0
START_ITERATION = 0

#  GLOBAL VARIABLES
Dataset           = None
HOGExtractFactory = None
LSTMModel         = None


########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
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

def CreateTrainData(imsHOGFeatures, defaultBboxs, bboxs):
    features = []
    preds    = []
    gts      = []
    for idx, imHOGFeatures in enumerate(imsHOGFeatures):
        bbox = bboxs[idx]

        feature     = []
        for imHOGFeature in imHOGFeatures:
            feature.append(numpy.asarray(imHOGFeature[0], dtype = 'float32'))
        feature = numpy.concatenate(feature, axis = 0)
        pred, gt = createGroundTruth(defaultBboxs, bbox)

        features.append(feature)
        preds.append(pred)
        gts.append(gt)

    features = numpy.asarray(features, dtype = 'float32')
    preds    = numpy.asarray(preds, dtype='float32')
    gts      = numpy.asarray(gts, dtype='float32')

    return features, preds, gts


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

def TrainModel():
    global Dataset, LSTMModel, HOGExtractFactory

    # Create startStateS | startStateC
    startStateS = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    startStateC = numpy.zeros((LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype = 'float32')

    # Plot training cost
    iterVisualize = []
    costVisualize = []
    plt.ion()
    data, = plt.plot(iterVisualize, costVisualize)
    plt.axis([13000, 13010, 0, 10])

    # Load model
    file = open(SAVE_PATH % (0, 13000))
    LSTMModel.LoadModel(file)
    file.close()
    print ('Load model !')

    # Train each Folder in Train folder
    Dataset.DataOpts['data_phase'] = 'train'
    allFolderNames = Dataset.GetAllFolderNames()
    defaultBboxs   = None
    iter  = 13001
    costs = []
    for folderName in allFolderNames:
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_folder_type'] = 'gt'
        allObjectIds = Dataset.GetAllObjectIds()

        for epoch in range(NUM_EPOCH):
            for objectId in allObjectIds:
                if epoch == 0 and objectId < 14:
                    continue

                Dataset.DataOpts['data_object_id'] = objectId
                imsPath, bboxs = Dataset.GetSequenceBy()


                DrawImage(imsPath, bboxs)














                if (imsPath.__len__() < NUM_TRUNCATE):
                    continue

                NUM_BATCH = (imsPath.__len__() - 1) // NUM_TRUNCATE
                S = startStateS;    C = startStateC
                for batchId in range(NUM_BATCH):
                    imsPathBatch = imsPath[batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE + 1]
                    bboxsBatch   = bboxs[batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE + 1]
                    imsHOGFeatures, defaultBboxs = HOGExtractFactory.ExtractFeature(imPaths = imsPathBatch, defaultBboxs = defaultBboxs)

                    features, preds, gts = CreateTrainData(imsHOGFeatures, defaultBboxs, bboxsBatch)
                    features = features[0 : NUM_TRUNCATE]
                    preds    = preds[1 : ]
                    gts      = gts[1 : ]

                    featuresgts = getFeatures(features, preds)

                    preds = preds.reshape((NUM_TRUNCATE, preds.shape[1], 1))
                    gts   = gts.reshape((NUM_TRUNCATE, preds.shape[1], 6 * 4))

                    print ('Load batch ! Done !')

                    test = True
                    for ft in featuresgts:
                        if ft.__len__() == 0:
                            test = False
                            break
                    if test == False:
                        continue

                    for k in range(10):
                        featuresgt = getRandomFeatures(featuresgts)

                        iter += 1
                        cost, newS, newC = LSTMModel.TrainFunc(featuresgt, features, preds, gts, S, C)
                        costs.append(cost)

                        if iter % DISPLAY_FREQUENCY == 0:
                            print ('Epoch = %d, iteration = %d, cost = %f. ObjectId = %d' % (epoch, iter, numpy.mean(costs), objectId))
                            iterVisualize.append(iter)
                            costVisualize.append(numpy.mean(costs))

                            data.set_xdata(numpy.append(data.get_xdata(), iterVisualize[-1]))
                            data.set_ydata(numpy.append(data.get_ydata(), costVisualize[-1]))
                            plt.axis([13000, iterVisualize[-1], 0, 10])
                            plt.draw()
                            plt.pause(0.05)
                            costs = []

                        if iter % SAVE_FREQUENCY == 0:
                            file = open(SAVE_PATH % (epoch, iter), 'wb')
                            LSTMModel.SaveModel(file)
                            file.close()
                            print ('Save model !')

                    S = newS;       C = newC

def DrawImage(imsPath, bboxs):
    for imPath, bbox in zip(imsPath, bboxs):
        rawIm = cv2.imread(imPath)


        fig, ax = plt.subplots(1)
        ax.imshow(rawIm)

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

def getRandomFeatures(features):
    featuresgt = numpy.zeros((features.__len__(), features[0][0].shape[0]), dtype='float32')
    for idx in range(features.__len__()):
        featuresgt[idx] = features[idx][numpy.random.randint(features[idx].__len__())]

    return featuresgt

if __name__ == '__main__':
    LoadDataset()
    CreateLSTMModel()
    CreateHOGExtractFactory()
    TrainModel()