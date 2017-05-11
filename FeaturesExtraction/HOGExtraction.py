import Utils.ImageHelper as ImageHelper
import cv2
import numpy
import math

class HOGExtraction():
    def __init__(self):
        # Get all information from config and save to its class
        self.FeatureOpts = {}
        self.FeatureOpts['num_scales_per_octave'] = 3
        self.FeatureOpts['num_octaves']           = 4
        self.FeatureOpts['image_max_width']       = 1024
        self.FeatureOpts['image_type']            = 'gray'
        self.FeatureOpts['hog_win_size']          = (64, 64)
        self.FeatureOpts['hog_block_size']        = (64, 64)
        self.FeatureOpts['hog_block_stride']      = (16, 16)
        self.FeatureOpts['hog_win_stride']        = (16, 16)
        self.FeatureOpts['hog_cell_size']         = (8, 8)
        self.FeatureOpts['hog_num_bins']          = 9

        # Create HOG factory
        self.HOGFactory = cv2.HOGDescriptor(self.FeatureOpts['hog_win_size'],
                                            self.FeatureOpts['hog_block_size'],
                                            self.FeatureOpts['hog_block_stride'],
                                            self.FeatureOpts['hog_cell_size'],
                                            self.FeatureOpts['hog_num_bins'])

    def ExtractFeature(self,
                       imPaths,
                       defaultBboxs = None):
        allFeatures = []

        for imPath in imPaths:
            maxWidth = self.FeatureOpts['image_max_width']

            # Read image
            im = ImageHelper.ReadImage(imPath)

            # Convert image to gray
            imGray = ImageHelper.ConvertImage(im, self.FeatureOpts['image_type'])

            # Rescale image with width = maxWidth (1024 default)
            numChannel, imWidth, imHeight = ImageHelper.Size(imGray)
            # newImWidth, newImHeight       = ImageHelper.ScaleSize((imWidth, imHeight), (maxWidth, maxWidth))
            newImWidth, newImHeight = [maxWidth, maxWidth]

            # Create standard image which width always equals maxWidth
            imStd = ImageHelper.Resize(im      = imGray,
                                       newSize = (newImWidth, newImHeight))

            # ---------------- DIRTY CODE ----------------------------------
            if defaultBboxs is None:
                defaultBboxs = []
                minSizes  = []
                maxSizes  = []
                numScales = self.FeatureOpts['num_octaves'] * self.FeatureOpts['num_scales_per_octave']
                scaleFactor = (1. / 2) ** (1. / self.FeatureOpts['num_scales_per_octave'])
                windowWidth = self.FeatureOpts['hog_win_size'][0]
                for idx in range(numScales):
                    currentScaleFactor = 1 / scaleFactor ** (idx)
                    minSizes.append(windowWidth * currentScaleFactor * 1.0)
                    maxSizes.append(windowWidth * currentScaleFactor * 1.1)

                for idx in range(self.FeatureOpts['num_octaves'] * self.FeatureOpts['num_scales_per_octave']):
                    currentScaleFactor = scaleFactor ** (idx)
                    imResized = ImageHelper.Resize(im=imStd,
                                                   scaleFactor=(currentScaleFactor, currentScaleFactor))
                    _, imResizedWidth, imResizedHeight = ImageHelper.Size(imResized)

                    if (imResizedWidth < self.FeatureOpts['hog_win_size'][0] or imResizedHeight <
                        self.FeatureOpts['hog_win_size'][1]):
                        break

                    # ----------------- DIRTY CODE START ---------------------------
                    blockStride = self.FeatureOpts['hog_block_stride']
                    windowSize = self.FeatureOpts['hog_win_size']
                    defaultBbox = self.createHOGDefaultBBox(imResizedWidth,
                                                            imResizedHeight,
                                                            blockStride,
                                                            windowSize,
                                                            minSizes[idx] * currentScaleFactor,
                                                            maxSizes[idx] * currentScaleFactor)
                    defaultBboxs.append(numpy.asarray(defaultBbox, dtype = 'float32'))
                    # ----------------- DIRTY CODE END -----------------------------

                defaultBboxs = numpy.concatenate(defaultBboxs, axis = 0)

            # ---------------- DIRTY CODE (END) ----------------------------

            features    = []
            scaleFactor = (1. / 2) ** (1. / self.FeatureOpts['num_scales_per_octave'])
            for idx in range(self.FeatureOpts['num_octaves'] * self.FeatureOpts['num_scales_per_octave']):
                currentScaleFactor = scaleFactor ** (idx)
                imResized = ImageHelper.Resize(im          = imStd,
                                               scaleFactor = (currentScaleFactor, currentScaleFactor))
                _, imResizedWidth, imResizedHeight = ImageHelper.Size(imResized)

                if (imResizedWidth < self.FeatureOpts['hog_win_size'][0] or imResizedHeight < self.FeatureOpts['hog_win_size'][1]):
                    break

                featureSize, featureLength = self.CalculateFeatureSize(imResizedWidth, imResizedHeight)

                # Extract HOG feature and Reshape feature
                hogDescriptor = self.HOGFactory.compute(imResized, winStride = self.FeatureOpts['hog_win_stride'])
                numFeatures   = hogDescriptor.__len__() / featureLength
                hogDescriptor = hogDescriptor.reshape((numFeatures, featureLength))

                features.append([hogDescriptor, featureSize, featureLength])

            allFeatures.append(features)

        return allFeatures, defaultBboxs

    def createHOGDefaultBBox(self,
                             imWidth,
                             imHeight,
                             blockStride,
                             winSize,
                             minSize,
                             maxSize):
        aspectRatio = [1., 2., 1. / 2., 3., 1. / 3.]
        defaultBBoxs = []
        for h in range(0, imHeight - winSize[0] + 1, blockStride[0]):
            for w in range(0, imWidth - winSize[1] + 1, blockStride[1]):
                # Temp
                xmin  = h
                xmax  = h + winSize[0]
                ymin  = w
                ymax  = w + winSize[1]

                centerX = (xmin + xmax) / 2.
                centerY = (ymin + ymax) / 2.

                DefaultBoxs = []
                # first prior: aspect_ratio = 1, size = min_size
                boxWidth = boxHeight = minSize
                # cx
                cx = centerX / imWidth
                # cy
                cy = centerY / imHeight
                # width
                width  = boxWidth / imWidth
                # height
                height = boxHeight / imHeight
                DefaultBoxs.append([cx, cy, width, height])

                # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                boxWidth = boxHeight = math.sqrt(minSize * maxSize);
                # cx
                cx = centerX / imWidth
                # cy
                cy = centerY / imHeight
                # width
                width = boxWidth / imWidth
                # height
                height = boxHeight / imHeight
                DefaultBoxs.append([cx, cy, width, height])

                for ar in aspectRatio:
                    if ar == 1:
                        continue

                    boxWidth = minSize * math.sqrt(ar)
                    boxHeight = minSize / math.sqrt(ar)

                    # cx
                    cx = centerX / imWidth
                    # cy
                    cy = centerY / imHeight
                    # width
                    width = boxWidth / imWidth
                    # height
                    height = boxHeight / imHeight
                    DefaultBoxs.append([cx, cy, width, height])

                defaultBBoxs.append(DefaultBoxs)

        return defaultBBoxs


    def CalculateFeatureSize(self, width, height):
        blockSize   = self.FeatureOpts['hog_block_size']
        cellSize    = self.FeatureOpts['hog_cell_size']
        blockStride = self.FeatureOpts['hog_block_stride']

        blockShape    =  (blockSize[0] / cellSize[0], blockSize[1] / cellSize[1])
        featureLength = self.FeatureOpts['hog_num_bins'] * blockShape[0] * blockShape[1]
        featureSize = ((width  - blockSize[0]) / blockStride[0] + 1,
                       (height - blockSize[1]) / blockStride[1] + 1)
        return featureSize, featureLength











