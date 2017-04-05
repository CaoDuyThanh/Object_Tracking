import os
from DataHelper import DatasetHelper
from FileHelper import *

class MOTDataHelper(DatasetHelper):

    def __init__(self,
                 datasetPath = None
                 ):
        # Check parameters
        CheckNotNone(datasetPath, 'datasetPath'); CheckPathExist(datasetPath)

        # Set parameters
        self.DatasetPath  = datasetPath

        # Config train | test dataset
        self.TrainPath    = self.DatasetPath + 'train/'
        self.TestPath     = self.DatasetPath + 'test/'
        self.TrainFolders = GetAllSubfoldersPath(self.TrainPath)
        self.TestFolders  = GetAllSubfoldersPath(self.TestPath)

        # Load data
        self.loadTrainFile()
        self.loadTestFile()

    # ------------------------  CHECK BBOX IN RANGE OF IMAGE  ---------------------------------------------
    def checkBBox(self, topLeftX, topLeftY, width, height, realWidth, realHeight):
        topLeftX = max(0, topLeftX)
        topLeftY = max(0, topLeftY)
        width    = min(realWidth, topLeftX + width) - topLeftX
        height   = min(realHeight, topLeftY + height) - topLeftY

        return [topLeftX, topLeftY, width, height]


    # ------------------------  LOAD TRAIN | VALID | TEST FILES--------------------------------------------
    def getDataFromOneFolder(self, path):
        # Check folder
        CheckPathExist(path)

        # Config img1 folder exist
        img1Folder = path + 'img1/';      CheckPathExist(img1Folder)

        # Create empty dictionary stored all data
        Data  = dict()

        # Get image info
        seqInfoPath = path + 'seqinfo.ini'
        seqInfo     = ReadFileIni(seqInfoPath)
        imageInfo   = dict()
        imageInfo['imagewidth'] = float(seqInfo['Sequence']['imWidth'])
        imageInfo['imageheight'] = float(seqInfo['Sequence']['imHeight'])
        Data['imageinfo'] = imageInfo

        # Read det.txt
        detFolder  = path + 'det/';
        if CheckPathExist(detFolder, throwError = False):
            Frames      = dict()
            FramesPath  = dict()
            ObjectId    = dict()
            detFilePath = detFolder + 'det.txt';    CheckFileExist(detFilePath)
            allDets     = ReadFile(detFilePath)
            for det in allDets:
                data = det.split(',')

                frameId  = float(data[0])  # Which frame object appears
                objectId = float(data[1])  # Number identifies that object as belonging to a tragectory by unique ID
                topLeftX = float(data[2])  # Topleft corner of bounding box (x)
                topLeftY = float(data[3])  # Topleft corner of bounding box (y)
                width    = float(data[4])  # Width of the bounding box
                height   = float(data[5])  # Height of the bounding box
                isIgnore = float(data[6])  # Flag whether this particular instance is ignored in the evaluation
                type     = float(data[7])  # Identify the type of object
                                            #     Label                ID
                                            # Pedestrian                1
                                            # Person on vehicle         2
                                            # Car                       3
                                            # Bicycle                   4
                                            # Motorbike                 5
                                            # Non motorized vehicle     6
                                            # Static person             7
                                            # Distrator                 8
                                            # Occluder                  9
                                            # Occluder on the ground   10
                                            # Occluder full            11
                                            # Reflection               12

                if frameId not in Frames:
                    Frames[frameId] = dict()
                topLeftX, topLeftY, width, height = self.checkBBox(topLeftX, topLeftY, width, height, imageInfo['imagewidth'], imageInfo['imageheight'])
                Frames[frameId][objectId] = [topLeftX, topLeftY, width, height, isIgnore, type]

                if objectId not in ObjectId:
                    ObjectId[objectId] = frameId

                if frameId not in FramesPath:
                    FramesPath[frameId] = os.path.abspath(os.path.join(img1Folder, '{0:06}'.format(frameId) + '.jpg'))

            DetData = dict()
            DetData['frames']     = Frames
            DetData['objectid']   = ObjectId
            DetData['framespath'] = FramesPath
            Data['det'] = DetData

        # Read gt.txt
        gtFolder = path + 'gt/';
        if CheckPathExist(gtFolder, throwError = False):
            Frames     = dict()
            FramesPath = dict()
            ObjectId   = dict()
            gtFilePath  = gtFolder + 'gt.txt';      CheckFileExist(gtFilePath)
            allGts = ReadFile(gtFilePath)
            for gt in allGts:
                data = gt.split(',')

                frameId  = int(data[0])      # Which frame object appears
                objectId = int(data[1])      # Number identifies that object as belonging to a tragectory by unique ID
                topLeftX = int(data[2])      # Topleft corner of bounding box (x)
                topLeftY = int(data[3])      # Topleft corner of bounding box (y)
                width    = int(data[4])      # Width of the bounding box
                height   = int(data[5])      # Height of the bounding box
                isIgnore = int(data[6])      # Flag whether this particular instance is ignored in the evaluation
                type     = int(data[7])      # Identify the type of object
                                                #     Label                ID
                                                # Pedestrian                1
                                                # Person on vehicle         2
                                                # Car                       3
                                                # Bicycle                   4
                                                # Motorbike                 5
                                                # Non motorized vehicle     6
                                                # Static person             7
                                                # Distrator                 8
                                                # Occluder                  9
                                                # Occluder on the ground   10
                                                # Occluder full            11
                                                # Reflection               12

                if frameId not in Frames:
                    Frames[frameId] = dict()
                topLeftX, topLeftY, width, height = self.checkBBox(topLeftX, topLeftY, width, height, imageInfo['imagewidth'], imageInfo['imageheight'])
                Frames[frameId][objectId] = [frameId, topLeftX, topLeftY, width, height, isIgnore, type]

                if objectId not in ObjectId:
                    ObjectId[objectId] = frameId

                if frameId not in FramesPath:
                    FramesPath[frameId] = os.path.abspath(os.path.join(img1Folder, '{0:06}'.format(frameId) + '.jpg'))

            GtData = dict()
            GtData['frames']     = Frames
            GtData['objectid']   = ObjectId
            GtData['framespath'] = FramesPath
            Data['gt'] = GtData

        return Data

    def loadTrainFile(self):
        self.TrainData = dict()
        self.TrainFoldersName = []
        for trainFolder in self.TrainFolders:
            folderName = trainFolder.split('/')[-2]
            self.TrainFoldersName.append(folderName)
            self.TrainData[folderName] = self.getDataFromOneFolder(trainFolder)

    def loadTestFile(self):
        self.TestData        = dict()
        self.TestFoldersName = []
        for testFolder in self.TestFolders:
            folderName = testFolder.split('/')[-2]
            self.TestFoldersName.append(folderName)
            self.TestData[folderName] = self.getDataFromOneFolder(testFolder)

    # -----------------------------------------------------------------------------------------------------

    def GetFramesPath(self,
                      folderName,
                      startFrame = 1,           # None = Start from the first frame
                      endFrame   = 10000000     # None = To the end of frame
                      ):
        frameId = startFrame
        framesPath = []
        while (frameId < endFrame):
            if frameId not in self.TrainData[folderName]['gt']['framespath']:
                break;
            framesPath.append(self.TrainData[folderName]['gt']['framespath'][frameId])
            frameId += 1

        return framesPath

    def GetSequenceBy(self, folderName, objectId):
        data = self.TrainData[folderName]
        Frames     = data['gt']['frames']
        ObjectId   = data['gt']['objectid']
        frameStart = ObjectId[objectId]
        sequence = []
        while frameStart < Frames.__len__():
            currentFrame = Frames[frameStart]
            if objectId in currentFrame:
                sequence.append(currentFrame[objectId])
            else:
                break;
            frameStart += 1

        return sequence


    def NextTrainBatch(self): raise NotImplementedError

    def NextValidBatch(self): raise NotImplementedError

    def NextTestBatch(self, batchSize): raise NotImplementedError