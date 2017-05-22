import numpy
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import thread
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
# SAVE DATA TRAINING
FILE_RECORD = 'record.pkl'

# TRAIN | VALID | TEST RATIO
TRAIN_RATIO = 0.8
VALID_RATIO = 0.05
TEST_RATIO  = 0.15

# TRAINING HYPER PARAMETER
BATCH_SIZE         = 1
NUM_EPOCH          = 10
DISPLAY_FREQUENCY  = 10;     INFO_DISPLAY = 'Epoch = %d, iteration = %d, cost = %f, costPos = %f, costNeg = %f'
SAVE_FREQUENCY     = 500
VALIDATE_FREQUENCY = 100

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = 8
NUM_HIDDEN        = 512
INPUTS_SIZE       = [5461 + 5461]
OUTPUTS_SIZE      = [5461]
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
START_EPOCH     = 2
START_ITERATION = 31500

#  GLOBAL VARIABLES
Dataset           = None
FeatureFactory    = None
DefaultBboxs      = None
BoxsVariances     = None
LSTMModel         = None
IsPause           = False


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
    FeatureFactory = SSD512FeaExtraction()
    FeatureFactory.LoadCaffeModel('../Models/SSD_512x512/VOC0712/deploy.prototxt',
                                  '../Models/SSD_512x512/VOC0712/VGG_coco_SSD_512x512_iter_360000.caffemodel')
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

def GetBatchMetaData(batchObjectIds):
    global Dataset

    imsPathBatch = []
    bboxsBatch = []
    maxSequence = 0
    for oneObjectId in batchObjectIds:
        folderName = oneObjectId[0]
        objectId = oneObjectId[1]
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

def GetBatchData(imsPathBatch,
                 bboxsBatch,
                 sequenceIdx):
    # Extract features of batch
    inputBatch = numpy.zeros((BATCH_SIZE, NUM_TRUNCATE, 3, 512, 512), dtype='float32')
    heatmapXBatch = numpy.zeros((BATCH_SIZE, NUM_TRUNCATE, 5461, 1), dtype='float32')
    heatmapYBatch = numpy.zeros((BATCH_SIZE, NUM_TRUNCATE, 5461, 1), dtype='float32')

    for objectIdx in range(BATCH_SIZE):
        # Get sequence
        imsPathSequence = imsPathBatch[objectIdx][sequenceIdx * NUM_TRUNCATE: (sequenceIdx + 1) * NUM_TRUNCATE + 1]
        bboxsSequence   = bboxsBatch  [objectIdx][sequenceIdx * NUM_TRUNCATE: (sequenceIdx + 1) * NUM_TRUNCATE + 1]

        # Extract feature and prepare bounding box before training....
        inputSequence   = ReadImages(imsPath = imsPathSequence, batchSize = NUM_TRUNCATE + 1)  # Extract sequence features
        heatmapSequence = CreateHeatmapSequence(DefaultBboxs, bboxsSequence)

        inputSequence    = inputSequence  [0 : NUM_TRUNCATE]
        heatmapXSequence = heatmapSequence[0 : NUM_TRUNCATE]
        heatmapYSequence = heatmapSequence[1 : NUM_TRUNCATE + 1]

        inputBatch[objectIdx, :, :, :, :] = inputSequence
        heatmapXBatch[objectIdx, :, :, :] = heatmapXSequence
        heatmapYBatch[objectIdx, :, :, :] = heatmapYSequence
    inputBatch = inputBatch.reshape((BATCH_SIZE * NUM_TRUNCATE, 3, 512, 512))

    return inputBatch, heatmapXBatch, heatmapYBatch

########################################################################################################################
#                                                                                                                      #
#    VALID LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def ValidModel(ValidData):
    global Dataset, LSTMModel, FeatureFactory, DefaultBboxs, BoxsVariances
    print ('---------------------------------------- VALID MODEL -----------------------------------------------------')

    # Create startStateS | startStateC
    startStateS = numpy.zeros((BATCH_SIZE, LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype='float32')
    startStateC = numpy.zeros((BATCH_SIZE, LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype='float32')

    S = startStateS;        C = startStateC

    # Train each folder in train folder
    iter  = 0
    CostsValid = []
    costs      = []
    costsPos   = []
    costsNeg   = []
    epoch = 0
    # Training start from here..........................................................................................
    numBatchObjectIds = ValidData.__len__() // BATCH_SIZE
    for batchObjectIdx in range(numBatchObjectIds):
        batchObjectIds = ValidData[batchObjectIdx * BATCH_SIZE: (batchObjectIdx + 1) * BATCH_SIZE]

        # Print information
        print ('    Load metadata of batch of objectIds...................')
        for oneObjectId in batchObjectIds:
            folderName = oneObjectId[0]
            objectId   = oneObjectId[1]
            print ('        Folder name = %s        ObjectId = %d' % (folderName, objectId))

        # Get information of batch of object ids
        imsPathBatch, bboxsBatch, maxSequence = GetBatchMetaData(batchObjectIds=batchObjectIds)

        # Reset status at the beginning of each sequence
        S = startStateS;
        C = startStateC;

        # Iterate sequence of batch
        numSequence = (maxSequence - 1) // NUM_TRUNCATE
        for sequenceIdx in range(numSequence):
            inputBatch, heatmapXBatch, heatmapYBatch = GetBatchData(imsPathBatch = imsPathBatch,
                                                                    bboxsBatch   = bboxsBatch,
                                                                    sequenceIdx  = sequenceIdx)
            print ('        Load batch. Done !')

            iter += 1
            result = LSTMModel.ValidFunc(inputBatch,
                                         heatmapXBatch,
                                         heatmapYBatch,
                                         S, C)
            cost    = result[0]
            newS    = result[1: 1 + BATCH_SIZE]
            newC    = result[1 + BATCH_SIZE: 1 + 2 * BATCH_SIZE]
            costPos = result[1 + 2 * BATCH_SIZE]
            costNeg = result[2 + 2 * BATCH_SIZE]
            costs.append(cost)
            costsPos.append(costPos)
            costsNeg.append(costNeg)
            print ('        Valid mini sequence in a batch ! Done !')

            if iter % DISPLAY_FREQUENCY == 0:
                # Print information of current training in progress
                print ('        ' + INFO_DISPLAY % (epoch, iter, numpy.mean(costs), numpy.mean(costsPos), numpy.mean(costsNeg)))

                # Plot result in progress
                CostsValid.append(numpy.mean(costs))

                # Empty costs for next visualization
                costs = []
                costsPos = []
                costsNeg = []

            S = numpy.asarray(newS, dtype='float32');
            C = numpy.asarray(newC, dtype='float32')

    print ('----------------------------------------------------------------------------------------------------------')
    return numpy.mean(CostsValid)


########################################################################################################################
#                                                                                                                      #
#    TRAIN LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def TrainModel():
    global Dataset, LSTMModel, FeatureFactory, DefaultBboxs, IsPause

    # Get all data and devide into TRAIN | VALID | TEST set
    Dataset.DataOpts['data_phase'] = 'train'
    allFolderNames = Dataset.GetAllFolderNames()
    allData = []
    for folderName in allFolderNames:
        Dataset.DataOpts['data_folder_name'] = folderName
        Dataset.DataOpts['data_folder_type'] = 'gt'
        allObjectIds = Dataset.GetAllObjectIds()
        for objectId in allObjectIds:
            allData.append([folderName, objectId])
    # Shuffle data
    random.seed(123456)
    random.shuffle(allData)

    # Divide into TRAIN|VALID|TEST set
    TrainData = allData[0 : int(math.floor(allData.__len__() * TRAIN_RATIO))]
    ValidData = allData[int(math.floor(allData.__len__() * TRAIN_RATIO)) : int(math.floor(allData.__len__() * (TRAIN_RATIO + VALID_RATIO)))]
    TestData  = allData[int(math.floor(allData.__len__() * (TRAIN_RATIO + VALID_RATIO))) :]

    # Load previous data record
    IterTrainRecord = []
    CostTrainRecord = []
    IterValidRecord = []
    CostValidRecord = []
    if CheckFileExist(FILE_RECORD, throwError = False):
        file = open(FILE_RECORD)
        IterTrainRecord = pickle.load(file)
        CostTrainRecord = pickle.load(file)
        IterValidRecord = pickle.load(file)
        CostValidRecord = pickle.load(file)
        file.close()
        print ('Load previous record !')

    # Plot training cost
    plt.ion()
    data, = plt.plot(IterTrainRecord, CostTrainRecord)

    # Load modelalidModel
    if CheckFileExist(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION),
                      throwError = False) == True:
        file = open(LOAD_MODEL_PATH % (START_EPOCH, START_ITERATION))
        LSTMModel.LoadModel(file)
        file.close()
        print ('Load model !')

    # Create startStateS | startStateC
    startStateS = numpy.zeros((BATCH_SIZE, LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype='float32')
    startStateC = numpy.zeros((BATCH_SIZE, LSTMModel.Net.LayerOpts['lstm_num_hidden'],), dtype='float32')
    S = startStateS; C = startStateC

    # Train each folder in train folder
    iter     = START_ITERATION + 1
    objectTrain = 0
    costs    = []
    costsPos = []
    costsNeg = []

    # Find id of objectId to start
    for startId in range(TrainData.__len__()):
        if TrainData[startId][0] == 'MOT16-09' and TrainData[startId][1] == 7:
            break
    objectTrain += START_EPOCH * TrainData.__len__() // BATCH_SIZE
    startId     += objectTrain

    print ('objectTrain = %d' % (objectTrain))

    # Training start from here..........................................................................................
    for epoch in xrange(START_EPOCH, NUM_EPOCH):
        numBatchObjectIds = TrainData.__len__() // BATCH_SIZE
        for batchObjectIdx in range(numBatchObjectIds):
            # Get batch of object id
            batchObjectIds = TrainData[batchObjectIdx * BATCH_SIZE: (batchObjectIdx + 1) * BATCH_SIZE]

            # Print information
            print ('Load metadata of batch of objectIds...................')
            for oneObjectId in batchObjectIds:
                folderName = oneObjectId[0]
                objectId   = oneObjectId[1]
                print ('    Folder name = %s        ObjectId = %d' % (folderName, objectId))

            if epoch == START_EPOCH and objectTrain < startId:
                objectTrain += 1
                continue

            # Validate model
            if objectTrain != 0 and objectTrain % VALIDATE_FREQUENCY == 0:
                costValid = ValidModel(ValidData = ValidData)

                IterValidRecord.append(iter)
                CostValidRecord.append(costValid)
                print (costValid)

            # Get information of batch of object ids
            imsPathBatch, bboxsBatch, maxSequence = GetBatchMetaData(batchObjectIds = batchObjectIds)

            # Reset status at the beginning of each sequence
            S = startStateS; C = startStateC;

            # Iterate sequence of batch
            numSequence = (maxSequence - 1) // NUM_TRUNCATE
            for sequenceIdx in range(numSequence):
                if (IsPause):
                    print ('Pause training process....................')
                    while (1):
                        input   = raw_input('Enter anything to resume...........')
                        if input == 'r':
                            IsPause = False
                            break;
                    print ('Resume !')

                inputBatch, heatmapXBatch, heatmapYBatch = GetBatchData(imsPathBatch = imsPathBatch,
                                                                        bboxsBatch   = bboxsBatch,
                                                                        sequenceIdx  = sequenceIdx)

                iter += 1
                result = LSTMModel.TrainFunc(inputBatch,
                                             heatmapXBatch,
                                             heatmapYBatch,
                                             S, C)
                cost = result[0]
                newS = result[1 : 1 + BATCH_SIZE]
                newC = result[1 + BATCH_SIZE : 1 + 2 * BATCH_SIZE]
                costPos = result[1 + 2 * BATCH_SIZE]
                costNeg = result[2 + 2 * BATCH_SIZE]
                costs.append(cost)
                costsPos.append(costPos)
                costsNeg.append(costNeg)
                print ('    Train mini sequence in a batch ! Done !')

                if iter % DISPLAY_FREQUENCY == 0:
                    # Print information of current training in progress
                    print (INFO_DISPLAY % (epoch, iter, numpy.mean(costs), numpy.mean(costsPos), numpy.mean(costsNeg)))

                    # Plot result in progress
                    IterTrainRecord.append(iter)
                    CostTrainRecord.append(numpy.mean(costs))
                    data.set_xdata(numpy.append(data.get_xdata(), IterTrainRecord[-1]))
                    data.set_ydata(numpy.append(data.get_ydata(), CostTrainRecord[-1]))
                    yLimit = math.floor(numpy.max(CostTrainRecord) / 10) * 10 + 3
                    plt.axis([IterTrainRecord[-1000], IterTrainRecord[-1], 0, yLimit])
                    plt.draw()
                    plt.pause(0.05)

                    # Empty costs for next visualization
                    costs    = []
                    costsPos = []
                    costsNeg = []

                if iter % SAVE_FREQUENCY == 0:
                    # Save model
                    file = open(SAVE_PATH % (epoch, iter), 'wb')
                    LSTMModel.SaveModel(file)
                    file.close()
                    print ('Save model !')

                    # Save record
                    file = open(FILE_RECORD, 'wb')
                    pickle.dump(IterTrainRecord, file, 0)
                    pickle.dump(CostTrainRecord, file, 0)
                    pickle.dump(IterValidRecord, file, 0)
                    pickle.dump(CostValidRecord, file, 0)
                    file.close()
                    print ('Save record !')

                S = numpy.asarray(newS, dtype = 'float32');       C = numpy.asarray(newC, dtype = 'float32')

            objectTrain += 1

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


def WaitEvent(threadName):
    global IsPause

    print ('Start pause event')

    while (1):
        input = raw_input()
        if input == 'p':
            IsPause = True

def CreatePauseEvent():
    try:
        thread.start_new_thread(WaitEvent, ('Thread wait',))
    except:
        print ('Error: unable to start thread')

if __name__ == '__main__':
    CreatePauseEvent()
    LoadDataset()
    CreateSSDExtractFactory()
    CreateLSTMModel()
    TrainModel()
