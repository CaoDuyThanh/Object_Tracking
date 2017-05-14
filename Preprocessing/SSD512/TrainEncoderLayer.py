import io
import matplotlib.pyplot as plt
import skimage.transform
import urllib
from Utils.FileHelper import *
from os import listdir
from os.path import abspath, join, isdir, isfile
from Models.CustomSSD.SSD512CustomModel import *
from Utils.FilterHelper import *
from PIL import Image


BATCH_SIZE = 1
NUM_EPOCH  = 4
NUM_SHOW   = 100
NUM_SAVE   = 1000
SAVE_MODEL = 'ssd512_conv4_3_encode.pkl'
LAYER_NAME = 'conv4_3_norm_encode'
LAYER_W_SIZE = (512, 256)
IMG_SIZE     = (16, 32)

# Model
customSSDModel = None

def LoadMode():
    global customSSDModel
    customSSDModel = SSD512CustomModel(batchSize = BATCH_SIZE)
    customSSDModel.LoadCaffeModel('../../Models/SSD_512x512/VOC0712/deploy.prototxt',
                                  '../../Models/SSD_512x512/VOC0712/VGG_coco_SSD_512x512_iter_360000.caffemodel')

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
        im = skimage.transform.resize(im, (512, 512), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (512, 512), preserve_range=True)

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

    file = open('ssd512_conv4_3_encode.pkl')
    customSSDModel.LoadLayers(file=file, layerNames=['conv4_3_norm_encode'])
    file.close()
    print('Load layer!')

    # # Show all fileters from model
    # image = Image.fromarray(tile_raster_images(
    #     X=customSSDModel.Net.Layer['conv4_3_norm_encode'].W.get_value(borrow=False).reshape((512, 256)).T,
    #     img_shape=(16, 32), tile_shape=(10, 10),
    #     tile_spacing=(1, 1)))
    # image.save('filters_corruption_30_0.png')

    # Load model
    if CheckFileExist(SAVE_MODEL,
                      throwError=False) == True:
        file = open(SAVE_MODEL)
        customSSDModel.LoadLayers(file=file, layerNames=[LAYER_NAME])
        file.close()
        print('Load layer!')

    # Show all fileters from model
    image = Image.fromarray(tile_raster_images(
        X=customSSDModel.Net.Layer[LAYER_NAME].W.get_value(borrow = False).reshape(LAYER_W_SIZE).T,
        img_shape=IMG_SIZE, tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')

    allFilesPath = ListImage('/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/VOC2007/Train/JPEGImages')
    iter  = 0
    costs = []
    imBatch = numpy.zeros((BATCH_SIZE, 3, 512, 512), dtype='float32')
    idx     = 0
    for epoch in range(NUM_EPOCH):
        for url in allFilesPath:
            im = prep_image(url)
            if im.shape.__len__() != 3:
                continue

            # Convert RGB to BGR
            im = im[:, :, [2, 1, 0]]
            im = numpy.transpose(im, (2, 0, 1))
            im = numpy.reshape(im, (1, 3, 512, 512))

            if idx < BATCH_SIZE:
                imBatch[idx, :, :, :] = im
                idx += 1

            if idx == BATCH_SIZE:
                imBatch = numpy.asarray(imBatch - VGG_MEAN, dtype = 'float32')

                cost = customSSDModel.Conv4_3EncodeFunc(imBatch)
                costs.append(cost)

                if iter % NUM_SHOW == 0:
                    print('Epoch = %d, iteration = %d, cost = %f' % (epoch, iter, numpy.mean(costs)))
                    costs = []

                if iter % NUM_SAVE == 0:
                    # Save model
                    file = open(SAVE_MODEL, 'wb')
                    customSSDModel.SaveLayers(file=file, layerNames=[LAYER_NAME])
                    file.close()
                    print('Save model!')

                iter = (iter + 1) % allFilesPath.__len__()

                idx  = 0

        # Show all fileters from model
        image = Image.fromarray(tile_raster_images(
            X=customSSDModel.Net.Layer[LAYER_NAME].W.get_value(borrow=False).reshape(LAYER_W_SIZE).T,
            img_shape=IMG_SIZE, tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('filters_corruption_30.png')

if __name__ == '__main__':
    LoadMode()
    TrainEncoderLayer()