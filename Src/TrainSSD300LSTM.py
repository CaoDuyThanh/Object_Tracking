import numpy
import random
import thread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from FeaturesExtraction.SSD300FeaExtraction import *
from Utils.MOTDataHelper import *
from Models.LSTM.LSTMTrackingModel import *
from Utils.DefaultBox import *
from Utils.BBoxHelper import *
from Utils.FileHelper import *
from random import shuffle

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
FILE_RECORD = 'record.pkl'

# TRAIN | VALID | TEST RATIO
TRAIN_RATIO = 0.8
VALID_RATIO = 0.05
TEST_RATIO  = 0.15

# TRAINING HYPER PARAMETER
BATCH_SIZE         = 1
NUM_EPOCH          = 30
LEARNING_RATE      = 0.00001      # Starting learning rate
DISPLAY_FREQUENCY  = 20;         INFO_DISPLAY = 'LearningRate = %f, Epoch = %d, iteration = %d, cost = %f, costPos = %f, costNeg = %f'
SAVE_FREQUENCY     = 1000
VALIDATE_FREQUENCY = 10000
UPDATE_LEARNING_RATE = 5000000

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = (6, 1)
NUM_HIDDEN        = 2048
INPUTS_SIZE       = [256 + 4 + 256 + 4 + 256 + 4 + 256 + 4 + 256 + 4 + 256 + 4]
OUTPUTS_SIZE      = [1, 4]
SEQUENCE_TRAIN    = NUM_TRUNCATE * 2

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# SAVE MODEL PATH
SAVE_PATH       = '../Pretrained/SSD/LSTM_SSD_Epoch=%d_Iter=%d.pkl'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/SSD/LSTM_SSD_Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 13
START_ITERATION = 240000
START_FOLDER    = ''
START_OBJECTID  = 0

# STATE PATH
STATE_PATH = '../Pretrained/SSD/CurrentState.pkl'

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
                                  featureFactory      = FeatureFactory,
                                  featureXBatch = FeatureFactory.Net.Layer['features_reshape'].Output.reshape((BATCH_SIZE, numpy.sum(NUM_TRUNCATE), 1940, 256)))

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




########################################################################################################################
#                                                                                                                      #
#    VALID LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def ValidModel(ValidData):
    print ('\n')
    print ('------------------------- VALIDATE MODEL -----------------------------------------------------------------')
    print ('\n')

    global IsPause, \
           LSTMModel

    # Create startStateS | startStateC
    startStateC = numpy.zeros((BATCH_SIZE, LSTMModel.EncodeNet.LayerOpts['lstm_num_hidden'],), dtype='float32')
    startStateH = numpy.zeros((BATCH_SIZE, LSTMModel.EncodeNet.LayerOpts['lstm_num_hidden'],), dtype='float32')

    numBatchObjectIds = ValidData.__len__() // BATCH_SIZE
    epoch = 0
    iter  = 0
    costs = []
    costsPos = []
    costsNeg = []
    for batchObjectIdx in range(numBatchObjectIds):
        # Get batch of object id
        batchObjectIds = ValidData[batchObjectIdx * BATCH_SIZE: (batchObjectIdx + 1) * BATCH_SIZE]

        # Print information
        print ('    Load metadata of batch of objectIds...................')
        for oneObjectId in batchObjectIds:
            folderName = oneObjectId[0]
            objectId   = int(oneObjectId[1])
            print ('        Folder name = %s        ObjectId = %d' % (folderName, objectId))

        # Get information of batch of object ids
        imsPathBatch, bboxsBatch, maxSequence = GetBatchMetaData(batchObjectIds = batchObjectIds)

        if maxSequence < numpy.sum(NUM_TRUNCATE):
            continue

        # Reset status at the beginning of each sequence
        encodeC = startStateC;
        encodeH = startStateH;
        decodeC = startStateC;

        # Iterate sequence of batch
        numSequence = (maxSequence - NUM_TRUNCATE[1]) // (NUM_TRUNCATE[0])
        for sequenceIdx in range(numSequence):
            if (IsPause):
                print ('Pause validate process....................')
                while (1):
                    input = raw_input('Enter anything to resume...........')
                    if input == 'r':
                        IsPause = False
                        break;
                print ('Resume !')

            # Prepare bounding box before training....
            imsPathMiniBatch = []
            bboxsMiniBatch = []
            for i in range(BATCH_SIZE):
                imsPathMiniBatch.append(imsPathBatch[i][sequenceIdx * NUM_TRUNCATE[0] : (sequenceIdx + 1)* NUM_TRUNCATE[0] + NUM_TRUNCATE[1]])
                bboxsMiniBatch.append(bboxsBatch[i][sequenceIdx * NUM_TRUNCATE[0] : (sequenceIdx + 1)* NUM_TRUNCATE[0] + NUM_TRUNCATE[1]])
            FeatureEncodeXBatch, YsBatch, BboxYsBatch = GetBatchData(imsPathMiniBatch, bboxsMiniBatch)

            iter += 1
            result = LSTMModel.ValidFunc(FeatureEncodeXBatch,
                                         YsBatch,
                                         BboxYsBatch,
                                         encodeC,
                                         encodeH,
                                         decodeC)
            cost = result[0]
            newEncodeC = result[1: 1 + BATCH_SIZE]
            newEncodeH = result[1 + BATCH_SIZE: 1 + 2 * BATCH_SIZE]
            costPos = result[1 + 2 * BATCH_SIZE]
            costNeg = result[2 + 2 * BATCH_SIZE]
            costs.append(cost)
            costsPos.append(costPos)
            costsNeg.append(costNeg)
            print ('        Valid mini sequence in a batch ! Done !')

            if iter % DISPLAY_FREQUENCY == 0:
                # Print information of current training in progress
                print (INFO_DISPLAY % (LEARNING_RATE, epoch, iter, numpy.mean(costs), numpy.mean(costsPos), numpy.mean(costsNeg)))

            encodeC = numpy.asarray(newEncodeC, dtype='float32');
            encodeH = numpy.asarray(newEncodeH, dtype='float32');

    print ('\n')
    print ('------------------------- VALIDATE MODEL (DONE) ----------------------------------------------------------')
    print ('\n')

    return numpy.mean(costs), numpy.mean(costsPos), numpy.mean(costsNeg)


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
    TrainData = allData[0: int(math.floor(allData.__len__() * TRAIN_RATIO))]
    ValidData = allData[int(math.floor(allData.__len__() * TRAIN_RATIO)): int(math.floor(allData.__len__() * (TRAIN_RATIO + VALID_RATIO)))]
    TestData  = allData[int(math.floor(allData.__len__() * (TRAIN_RATIO + VALID_RATIO))):]

    # Sort Data based on its length
    TrainData = SortData(TrainData)
    ValidData = SortData(ValidData)

    # Load previous data record
    IterTrainRecord = []
    CostTrainRecord = []
    IterValidRecord = []
    CostValidRecord = []
    if CheckFileExist(FILE_RECORD, throwError=False):
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

    # Load Model
    if CheckFileExist(STATE_PATH,
                      throwError=False) == True:
        file = open(STATE_PATH)
        LSTMModel.LoadState(file)
        file.close()
        print ('Load state !')

    # Create startStateS | startStateC
    startStateC = numpy.zeros((BATCH_SIZE, LSTMModel.EncodeNet.LayerOpts['lstm_num_hidden'],), dtype = 'float32')
    startStateH = numpy.zeros((BATCH_SIZE, LSTMModel.EncodeNet.LayerOpts['lstm_num_hidden'],), dtype = 'float32')

    # Train each folder in train folder
    # objectTrain = 0

    # Find id of objectId to start
    # for startId in range(TrainData.__len__()):
    #     if TrainData[startId][0] == START_FOLDER and TrainData[startId][1] == START_OBJECTID:
    #         break
    # objectTrain += START_EPOCH * TrainData.__len__() // BATCH_SIZE
    # startId += objectTrain
    # print ('objectTrain = %d' % (objectTrain))

    costs    = []
    costsPos = []
    costsNeg = []
    iter         = START_ITERATION
    learningRate = LEARNING_RATE
    # Training start from here..........................................................................................
    for epoch in xrange(START_EPOCH, NUM_EPOCH):
        numBatchObjectIds = TrainData.__len__() // BATCH_SIZE
        allBatchObjectIds = range(numBatchObjectIds)
        shuffle(allBatchObjectIds)

        numBatchObjectTrained = 0
        for batchObjectIdx in allBatchObjectIds:
            # Get batch of object id
            batchObjectIds = TrainData[batchObjectIdx * BATCH_SIZE: (batchObjectIdx + 1) * BATCH_SIZE]
            numBatchObjectTrained += 1

            # Print information
            print ('Load metadata of batch of objectIds (%d th in %d objects)...................' % (numBatchObjectTrained, len(allBatchObjectIds)))
            for oneObjectId in batchObjectIds:
                folderName = oneObjectId[0]
                objectId   = int(oneObjectId[1])
                print ('    Folder name = %s        ObjectId = %d' % (folderName, objectId))

            # Get information of batch of object ids
            imsPathBatch, bboxsBatch, maxSequence = GetBatchMetaData(batchObjectIds = batchObjectIds)

            if maxSequence < numpy.sum(NUM_TRUNCATE):
                continue

            # Reset status at the beginning of each sequence
            encodeC = startStateC;    encodeH = startStateH;
            decodeC = startStateC;

            # Iterate sequence of batch
            numSequence = (maxSequence - NUM_TRUNCATE[1]) // (NUM_TRUNCATE[0])
            for sequenceIdx in range(numSequence):
                if (IsPause):
                    print ('Pause training process....................')
                    while (1):
                        input = raw_input('Enter anything to resume...........')
                        if input == 'r':
                            IsPause = False
                            break;
                    print ('Resume !')

                if iter % UPDATE_LEARNING_RATE == 0:
                    learningRate /= 2

                # if iter <= START_ITERATION:
                #     iter += 1
                #     continue

                # Prepare bounding box before training....
                imsPathMiniBatch = []
                bboxsMiniBatch   = []
                for i in range(BATCH_SIZE):
                    imsPathMiniBatch.append(imsPathBatch[i][sequenceIdx * NUM_TRUNCATE[0] : (sequenceIdx + 1)* NUM_TRUNCATE[0] + NUM_TRUNCATE[1]])
                    bboxsMiniBatch.append(bboxsBatch[i][sequenceIdx * NUM_TRUNCATE[0] : (sequenceIdx + 1)* NUM_TRUNCATE[0] + NUM_TRUNCATE[1]])
                FeatureEncodeXBatch, YsBatch, BboxYsBatch = GetBatchData(imsPathMiniBatch, bboxsMiniBatch)

                # Draw(imsPathMiniBatch, YsBatch, BboxYsBatch)

                iter += 1
                result = LSTMModel.TrainFunc(learningRate,
                                             FeatureEncodeXBatch,
                                             YsBatch,
                                             BboxYsBatch,
                                             encodeC,
                                             encodeH,
                                             decodeC)
                cost       = result[0]
                newEncodeC = result[1                 : 1 +     BATCH_SIZE]
                newEncodeH = result[1 +     BATCH_SIZE: 1 + 2 * BATCH_SIZE]
                costPos = result[1 + 2 * BATCH_SIZE]
                costNeg = result[2 + 2 * BATCH_SIZE]
                costs.append(cost)
                costsPos.append(costPos)
                costsNeg.append(costNeg)
                print ('    Train mini sequence in a batch ! Done !')

                if iter % VALIDATE_FREQUENCY == 0:
                    costValid, posCostValid, negCostValid = ValidModel(ValidData = ValidData)
                    IterValidRecord.append(iter)
                    CostValidRecord.append(costValid)
                    print ('Validate model finished! Cost = %f, PosCost = %f, NegCost = %f' % (costValid, posCostValid, negCostValid))

                if iter % DISPLAY_FREQUENCY == 0:
                    # Print information of current training in progress
                    print (INFO_DISPLAY % (learningRate, epoch, iter, numpy.mean(costs), numpy.mean(costsPos), numpy.mean(costsNeg)))
                    print (ToString(costs))
                    print (ToString(costsPos))
                    print (ToString(costsNeg))

                    # Plot result in progress
                    IterTrainRecord.append(iter)
                    CostTrainRecord.append(numpy.mean(costs))
                    data.set_xdata(numpy.append(data.get_xdata(), IterTrainRecord[-1]))
                    data.set_ydata(numpy.append(data.get_ydata(), CostTrainRecord[-1]))
                    yLimit = math.floor(numpy.max(CostTrainRecord) / 10) * 10 + 10
                    plt.axis([IterTrainRecord[-1] - 10000, IterTrainRecord[-1], 0, yLimit])
                    plt.draw()
                    plt.pause(0.05)

                    # Empty costs for next visualization
                    costs = []
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

                    # Save state
                    file = open(STATE_PATH, 'wb')
                    LSTMModel.SaveState(file)
                    file.close()
                    print ('Save state !')

                encodeC = numpy.asarray(newEncodeC, dtype='float32');
                encodeH = numpy.asarray(newEncodeH, dtype='float32');
                # decodeC = numpy.asarray(newDecodeC, dtype='float32');

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

def ToString(arrs):
    str = ''
    for arr in arrs:
        str += '%f ' % arr
    return str

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