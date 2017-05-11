import math
import numpy

class BBoxOpts():
    def __init__(self):
        self.Opts = {}

        self.Opts['image_width']  = 800
        self.Opts['image_height'] = 600
        self.Opts['smin']         = 20
        self.Opts['smax']         = 90
        self.Opts['layer_sizes']  = None
        self.Opts['num_boxs']     = None
        self.Opts['offset']       = 0.5
        self.Opts['steps']        = None

class DefaultBBox():
    def __init__(self,
                 bboxOpts):
        # Save information to its layer
        self.ImageWidth  = bboxOpts.Opts['image_width']
        self.ImageHeight = bboxOpts.Opts['image_height']
        self.SMin        = bboxOpts.Opts['smin']
        self.SMax        = bboxOpts.Opts['smax']
        self.LayerSizes  = bboxOpts.Opts['layer_sizes']
        self.NumBoxs     = bboxOpts.Opts['num_boxs']
        self.Offset      = bboxOpts.Opts['offset']
        self.Steps       = bboxOpts.Opts['steps']
        # steps = [8, 16, 32, 64, 100, 300]

    def CreateDefaultBox(self):
        self.ListDefaultBoxes = []

        if self.Steps is None:
            self.Steps = []
            for layerSize in self.LayerSizes:
                step = round(self.ImageWidth / layerSize[0])
                self.Steps.append(step)

        step = int(math.floor((self.SMax - self.SMin) / (len(self.LayerSizes) - 2)))

        minSizes = []
        maxSizes = []
        for ratio in xrange(self.SMin, self.SMax + 1, step):
            minSizes.append(self.ImageWidth  *  ratio         / 100.)
            maxSizes.append(self.ImageHeight * (ratio + step) / 100.)
        minSizes = [self.ImageWidth * 10 / 100.] + minSizes
        maxSizes = [self.ImageWidth * 20 / 100.] + maxSizes

        for k, layerSize in enumerate(self.LayerSizes):
            layerWidth  = layerSize[0]
            layerHeight = layerSize[1]
            numbox      = self.NumBoxs[k]

            minSize = minSizes[k]
            maxSize = maxSizes[k]

            if numbox == 4:
                aspectRatio = [1., 2., 1. / 2.]
            else:
                aspectRatio = [1., 2., 1. / 2., 3., 1. / 3.]
            stepW = stepH = self.Steps[k]
            for h in range(layerHeight):
                for w in range(layerWidth):
                    centerX = (w + self.Offset) * stepW
                    centerY = (h + self.Offset) * stepH

                    DefaultBoxs = []
                    # first prior: aspect_ratio = 1, size = min_size
                    boxWidth = boxHeight = minSize
                    # xmin
                    xmin = (centerX - boxWidth / 2.)  / self.ImageWidth
                    # ymin
                    ymin = (centerY - boxHeight / 2.) / self.ImageHeight
                    # xmax
                    xmax = (centerX + boxWidth / 2.)  / self.ImageWidth
                    # ymax
                    ymax = (centerY + boxHeight / 2.) / self.ImageHeight
                    DefaultBoxs.append([xmin, ymin, xmax, ymax])

                    if maxSizes.__len__() > 0:
                        # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        boxWidth = boxHeight = math.sqrt(minSize * maxSize);
                        # xmin
                        xmin = (centerX - boxWidth / 2.)  / self.ImageWidth
                        # ymin
                        ymin = (centerY - boxHeight / 2.) / self.ImageHeight
                        # xmax
                        xmax = (centerX + boxWidth / 2.)  / self.ImageWidth
                        # ymax
                        ymax = (centerY + boxHeight / 2.) / self.ImageHeight
                        DefaultBoxs.append([xmin, ymin, xmax, ymax])

                    for ar in aspectRatio:
                        if ar == 1:
                            continue

                        boxWidth  = minSize * math.sqrt(ar)
                        boxHeight = minSize / math.sqrt(ar)

                        # xmin
                        xmin = (centerX - boxWidth / 2.)  / self.ImageWidth
                        # ymin
                        ymin = (centerY - boxHeight / 2.) / self.ImageHeight
                        # xmax
                        xmax = (centerX + boxWidth / 2.)  / self.ImageWidth
                        # ymax
                        ymax = (centerY + boxHeight / 2.) / self.ImageHeight
                        DefaultBoxs.append([xmin, ymin, xmax, ymax])

                    self.ListDefaultBoxes.append(DefaultBoxs)

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