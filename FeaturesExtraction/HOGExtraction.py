import Utils.ImageHelper as ImageHelper
import cv2

class HOGExtraction():
    def __init__(self,
                 config,
                 ):
        # Get all information from config and save to its class
        self.NumScalesPerOctave = config['hog_num_scales_per_octave']
        self.NumOctaves         = config['hog_num_octaves']
        self.CellSize           = config['hog_cell_size']
        self.BlockSize          = config['hog_block_size']
        self.NumBins            = config['hog_num_bins']

        # Need comment here
        self.Im    = []
        self.ImHOG = []

        # Create HOG factory
        self.HOGFactory = cv2.HOGDescriptor()

    def PushImage(self,
                  im):
        # Rescale image with width = 1024
        numChannel, imWidth, imHeight = ImageHelper.Size(im)
        newImWidth, newImHeight       = ImageHelper.ScaleSize((imWidth, imHeight), (1024, -1))

        # Create standard image which width always equals 1024
        im = ImageHelper.Resize(im, (newImWidth, newImHeight))

        scaleIm = (1. / 2) ^ (1. / self.NumScalesPerOctave)
        for idx in range(self.NumOctaves * self.NumScalesPerOctave):
            scaleFactor = scaleIm ^ (idx)
            imResized = ImageHelper.Resize(im, (scaleFactor, scaleFactor))
            self.Im.append(imResized)

            # Extract HOG features
            hogDescriptor = self.HOGFactory.compute(imResized)
            self.ImHOG.append(hogDescriptor)











