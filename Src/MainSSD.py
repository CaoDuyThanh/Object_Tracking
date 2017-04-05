import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import caffe
import math
from SSDModel import SSDModel
from DefaultBox import *

def ProposedMethod():
    ssdmodel = SSDModel()
    ssdmodel.LoadCaffeModel('../Models/SSD_300x300/VOC0712/deploy.prototxt', '../Models/SSD_300x300/VOC0712/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    TestModel(ssdmodel)

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

def TestModel(ssdmodel):
    image_urls = [ '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/003304.jpg',
                   '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/004932.jpg',
                   '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/004934.jpg',
                   '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/004835.jpg',
                   '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/004936.jpg',
                   '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/004937.jpg',
                   '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/004938.jpg',
                   '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/VOC/Test/VOCdevkit/VOC2007/JPEGImages/004939.jpg']

    VGG_MEAN = numpy.asarray([103.939, 116.779, 123.68])
    VGG_MEAN = numpy.reshape(VGG_MEAN, (1, 3, 1, 1))

    defaultBox = DefaultBox(300,
                            [[38, 38, 4],
                             [19, 19, 6],
                             [10, 10, 6],
                             [5, 5, 6],
                             [3, 3, 4],
                             [1, 1, 4]],
                            0.5)

    for url in image_urls:
        try:
            im = prep_image(url)
            if im.shape.__len__() != 3:
                continue
            rawIm = numpy.copy(im).astype('uint8')

            # Convert RGB to BGR
            im = im[:, :, [2, 1, 0]]
            im = numpy.transpose(im, (2, 0, 1))
            im = numpy.reshape(im, (1, 3, 300, 300))
            im = numpy.asarray(im - VGG_MEAN, dtype=numpy.float32)

            output = ssdmodel.TestNetwork(im)

            bestBox = defaultBox.Bbox(output[0], output[1])

            fig, ax = plt.subplots(1)
            ax.imshow(rawIm)

            for box in bestBox:
                rect = patches.Rectangle((box[0] * 300, box[1] * 300), (box[2] - box[0]) * 300, (box[3] - box[1]) * 300, linewidth=1, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)

            plt.show()
            plt.axis('off')

        except IOError:
            print('bad url: ' + url)
            continue


if __name__ == '__main__':
    ProposedMethod()
