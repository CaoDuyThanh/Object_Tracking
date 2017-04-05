import math
import numpy

class DefaultBox():
    def __init__(self,
                 minDim,
                 layerSizes,
                 offset):
        imageWidth = imageHeight = minDim
        self.ListBoxes = []
        self.MinDim = minDim

        smin = 20
        smax = 90
        step = int(math.floor((smax - smin) / (len(layerSizes) - 2)))
        minSizes = []
        maxSizes = []
        for ratio in xrange(smin, smax + 1, step):
            minSizes.append(minDim * ratio / 100.)
            maxSizes.append(minDim * (ratio + step) / 100.)
        minSizes = [minDim * 10 / 100.] + minSizes
        maxSizes = [minDim * 20 / 100.] + maxSizes
        steps = [8, 16, 32, 64, 100, 300]

        numLayers = layerSizes.__len__()
        for k, layerSize in enumerate(layerSizes):
            layerWidth    = layerSize[0]
            layerHeight   = layerSize[1]
            numbox        = layerSize[2]

            minSize = minSizes[k]
            maxSize = maxSizes[k]

            if numbox == 4:
                aspectRatio = [1., 2., 1. / 2.]
            else:
                aspectRatio = [1., 2., 1. / 2., 3., 1. / 3.]

            stepW = stepH = steps[k]

            for h in range(layerHeight):
                for w in range(layerWidth):
                    centerX = (w + offset) * stepW
                    centerY = (h + offset) * stepH
                    # first prior: aspect_ratio = 1, size = min_size
                    boxWidth = boxHeight = minSize
                    # xmin
                    xmin = (centerX - boxWidth / 2.) / imageWidth
                    # ymin
                    ymin = (centerY - boxHeight / 2.) / imageHeight
                    # xmax
                    xmax = (centerX + boxWidth / 2.) / imageWidth
                    # ymax
                    ymax = (centerY + boxHeight / 2.) / imageHeight

                    self.ListBoxes.append([xmin, ymin, xmax, ymax])

                    if maxSizes.__len__() > 0:
                        # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        boxWidth = boxHeight = math.sqrt(minSize * maxSize);
                        # xmin
                        xmin = (centerX - boxWidth / 2.) / imageWidth
                        # ymin
                        ymin = (centerY - boxHeight / 2.) / imageHeight
                        # xmax
                        xmax = (centerX + boxWidth / 2.) / imageWidth
                        # ymax
                        ymax = (centerY + boxHeight / 2.) / imageHeight

                        self.ListBoxes.append([xmin, ymin, xmax, ymax])

                    for ar in aspectRatio:
                        if ar == 1:
                            continue

                        boxWidth  = minSize * math.sqrt(ar)
                        boxHeight = minSize / math.sqrt(ar)

                        # xmin
                        xmin = (centerX - boxWidth / 2.) / imageWidth
                        # ymin
                        ymin = (centerY - boxHeight / 2.) / imageHeight
                        # xmax
                        xmax = (centerX + boxWidth / 2.) / imageWidth
                        # ymax
                        ymax = (centerY + boxHeight / 2.) / imageHeight

                        self.ListBoxes.append([xmin, ymin, xmax, ymax])

        # # Clip the prior's coordinate such that it is within [0, 1]
        # for box in self.ListBoxes:
        #     for idx in range(len(box)):
        #         box[idx] = min(max(box[idx], 0.), 1.)

    def Bbox(self,
             pred,
             boxes):
        bestBoxes = []
        for id, box in enumerate(boxes):
            archorBox = self.ListBoxes[id]
            archorXmin = archorBox[0]
            archorYmin = archorBox[1]
            archorXmax = archorBox[2]
            archorYmax = archorBox[3]
            cx = (archorXmin + archorXmax) / 2
            cy = (archorYmin + archorYmax) / 2
            w  = (archorXmax - archorXmin)
            h  = (archorYmax - archorYmin)

            offsetXmin = box[0]
            offsetYmin = box[1]
            offsetXmax = box[2]
            offsetYmax = box[3]

            cx = offsetXmin * 0.1 * w + cx
            cy = offsetYmin * 0.1 * h + cy
            w  = math.exp(offsetXmax * 0.2) * w
            h  = math.exp(offsetYmax * 0.2) * h

            if pred[id] != 0:
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