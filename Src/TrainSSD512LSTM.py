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
DISPLAY_FREQUENCY = 10;     INFO_DISPLAY = 'Epoch = %d, iteration = %d, cost = %f, folderName = %s, objectId = %d'
SAVE_FREQUENCY    = 150

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = 5
NUM_HIDDEN        = 512
INPUTS_SIZE       = [257]
OUTPUTS_SIZE      = [1]
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
START_EPOCH     = 0
START_ITERATION = 0

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
    FeatureFactory = SSD512FeaExtraction(batchSize = NUM_TRUNCATE + 1)
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
def CreateHeatmapSequence(defaultBboxs, groundTruths):
    heatmapSequence = []
    for idx, groundTruth in enumerate(groundTruths):
        heatmap = CreateHeatmap(defaultBboxs, groundTruth)
        heatmapSequence.append(heatmap)

    # Convert to numpy array
    heatmapSequence = numpy.asarray(heatmapSequence, dtype='float32')
    return heatmapSequence

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

    # Train each folder in train folder
    iter = START_ITERATION
    costs = []

    # Training start from here..........................................................................................
    Dataset.DataOpts['data_phase'] = 'train'
    allFolderNames = Dataset.GetAllFolderNames()
    for epoch in xrange(0, NUM_EPOCH):
        if epoch < START_EPOCH:     # We continue to train model from START_EPOCH
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
                if (imsPath.__len__() < NUM_TRUNCATE + 1):
                    continue

                # Else train the sequence ...................
                S = startStateS;    C = startStateC
                NUM_BATCH = (imsPath.__len__()) // NUM_TRUNCATE
                for batchId in range(NUM_BATCH):
                    print ('\r Load mini sequence !.....................')
                    # Get batch
                    imsPathSequence = imsPath[batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE + 1]
                    bboxsSequence   = bboxs  [batchId * NUM_TRUNCATE : (batchId + 1) * NUM_TRUNCATE + 1]

                    # Extract feature and prepare bounding box before training....
                    featureSequence = FeatureFactory.ExtractFeature(imsPath = imsPathSequence)   # Extract sequence features
                    heatmapSequence = CreateHeatmapSequence(DefaultBboxs, bboxsSequence)

                    featureSequence  = featureSequence[0: NUM_TRUNCATE]
                    heatmapXSequence = heatmapSequence[0: NUM_TRUNCATE]
                    heatmapYSequence = heatmapSequence[1: NUM_TRUNCATE + 1]
                    print ('\r Load mini sequence ! Done !')

                    print ('\r Train mini sequence !.....................')
                    iter += 1
                    cost, newS, newC = LSTMModel.TrainFunc(featureSequence,
                                                           heatmapXSequence,
                                                           heatmapYSequence,
                                                           S, C)
                    print ('\r Train mini sequence ! Done !')


                    costs.append(cost)

                    if iter % DISPLAY_FREQUENCY == 0:
                        # Print information of current training in progress
                        print (INFO_DISPLAY % (epoch, iter, numpy.mean(costs), folderName, objectId))

                        # Plot result in progress
                        iterVisualize.append(iter)
                        costVisualize.append(numpy.mean(costs))
                        data.set_xdata(numpy.append(data.get_xdata(), iterVisualize[-1]))
                        data.set_ydata(numpy.append(data.get_ydata(), costVisualize[-1]))
                        yLimit = math.floor(numpy.max(costVisualize) / 10) * 10 + 4
                        plt.axis([START_ITERATION, iterVisualize[-1], 0, yLimit])
                        plt.draw()
                        plt.pause(0.05)

                        # Empty costs for next visualization
                        costs = []

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