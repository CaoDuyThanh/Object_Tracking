import io
import matplotlib.pyplot as plt
import skimage.transform
import urllib
from os import listdir
from os.path import abspath, join, isdir, isfile
from Models.CustomSSD.SSD300CustomModel import *
from Utils.FilterHelper import *
from PIL import Image


NUM_EPOCH  = 10
NUM_SHOW   = 100
NUM_SAVE   = 1000
SAVE_MODEL = 'conv6_2_encode.pkl'


# Model
customSSDModel = None

def LoadMode():
    global customSSDModel
    customSSDModel = SSD300CustomModel()
    customSSDModel.LoadCaffeModel('../Models/SSD_300x300/VOC0712/deploy.prototxt',
                                  '../Models/SSD_300x300/VOC0712/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')

def ListImage(trainDir):
    sortedTrainFiles = sorted([join(abspath(trainDir), name) for name in listdir(trainDir)
                               if isfile(join(trainDir, name))])
    return sortedTrainFiles

def prep_image(url):
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    if im.shape.__len__() == 3:
        h, w, _ = im.shape
    else:
        if im.shape.__len__() == 4:
            h, w, _, _ = im.shape
        else:
            if im.shape.__len__() == 2:
                h, w = im.shape
    if h < w:
        im = skimage.transform.resize(im, (300, 300), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (300, 300), preserve_range=True)

    # Central crop to 224x224
    if im.shape.__len__() == 3:
        h, w, _ = im.shape
    else:
        if im.shape.__len__() == 4:
            h, w, _, _ = im.shape
        else:
            if im.shape.__len__() == 2:
                h, w = im.shape
    return im


def TrainEncoderLayer():
    global customSSDModel

    VGG_MEAN = numpy.asarray([103.939, 116.779, 123.68])
    VGG_MEAN = numpy.reshape(VGG_MEAN, (1, 3, 1, 1))

    # # Load model
    # file = open('conv6_2_encode.pkl')
    # customSSDModel.LoadLayers(file=file, layerNames=['conv6_2_encode'])
    # file.close()
    # print('Load layer!')
    #
    # # Show all fileters from model
    # image = Image.fromarray(tile_raster_images(
    #     X=customSSDModel.Net.Layer['conv6_2_encode'].W.get_value(borrow = False).reshape((512, 256)).T,
    #     img_shape=(32, 16), tile_shape=(10, 10),
    #     tile_spacing=(1, 1)))
    # image.save('filters_corruption_30.png')

    allFilesPath = ListImage('/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/VOC2007/Train/JPEGImages')
    iter  = 0
    costs = []
    for epoch in range(NUM_EPOCH):
        for url in allFilesPath:
            im = prep_image(url)
            if im.shape.__len__() != 3:
                continue

            # Convert RGB to BGR
            im = im[:, :, [2, 1, 0]]
            im = numpy.transpose(im, (2, 0, 1))
            im = numpy.reshape(im, (1, 3, 300, 300))
            im = numpy.asarray(im - VGG_MEAN, dtype=numpy.float32)

            cost = customSSDModel.Conv6_2EncodeFunc(im)
            costs.append(cost)

            if iter % NUM_SHOW == 0:
                print('Epoch = %d, iteration = %d, cost = %f' % (epoch, iter, numpy.mean(costs)))
                costs = []

            if iter % NUM_SAVE == 0:
                # Save model
                file = open(SAVE_MODEL, 'wb')
                customSSDModel.SaveLayers(file=file, layerNames=['conv6_2_encode'])
                file.close()
                print('Save model!')

            iter = (iter + 1) % allFilesPath.__len__()

if __name__ == '__main__':
    LoadMode()
    TrainEncoderLayer()