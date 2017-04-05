# Import library
import cv2
from VGG16Model import *

# Import modules
from Utils.MOTDataHelper import *
import matplotlib.pyplot as plt


# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'


#  GLOBAL VARIABLES
Dataset = None


def LoadDataset():
    global Dataset
    Dataset = MOTDataHelper(DATASET_PATH)

def ProposedMethod():
    vgg16model = VGG16Model()
    vgg16model.LoadCaffeModel('Models/VGG16.prototxt', 'Pretrained/VGG_ILSVRC_16_layers.caffemodel')
    TestModel(vgg16model)

import io
import skimage.transform
import urllib
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
        im = skimage.transform.resize(im, (256, w * 256 / h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h * 256 / w, 256), preserve_range=True)

    # Central crop to 224x224
    if im.shape.__len__() == 3:
        h, w, _ = im.shape
    else:
        if im.shape.__len__() == 4:
            h, w, _, _ = im.shape
        else:
            if im.shape.__len__() == 2:
                h, w = im.shape
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

    return im

def TestModel(vgg16model):
    index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')

    numpy.random.seed(50)
    numpy.random.shuffle(image_urls)
    image_urls = image_urls[2:20]

    VGG_MEAN = numpy.asarray([103.939, 116.779, 123.68])
    VGG_MEAN = numpy.reshape(VGG_MEAN, (1, 3, 1, 1))

    model = pickle.load(open('Pretrained/vgg_cnn_s.pkl'))
    CLASSES = model['synset words']

    for url in image_urls:
        try:
            im = prep_image(url)
            if im.shape.__len__() != 3:
                continue
            rawIm = numpy.copy(im).astype('uint8')

            # Convert RGB to BGR
            im = im[:, :, [2, 1, 0]]
            im = numpy.transpose(im, (2, 0, 1))
            im = numpy.reshape(im, (1, 3, 224, 224))
            im = numpy.asarray(im - VGG_MEAN, dtype=numpy.float32)

            prob = vgg16model.TestNetwork(im)
            top5 = numpy.argsort(prob[0])[-1:-6:-1]

            plt.figure()
            plt.imshow(rawIm)
            plt.axis('off')
            for n, label in enumerate(top5):
                plt.text(250, 70 + n * 20, '{}. {}'.format(n + 1, CLASSES[label]), fontsize=14)
        except IOError:
            print('bad url: ' + url)
            continue



if __name__ == '__main__':
    LoadDataset()
    ProposedMethod()