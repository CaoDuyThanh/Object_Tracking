import cv2
from FileHelper import *

def Size(im):
    """
    Get size of the image.

    Parameters
    ----------
    im: array_like
        image

    Returns
    -------
    NumChannel: int
        Number of channels of image (3 for rgb | 2 for grayscale)`

    Width: int
        Width of image

    Height: int
        Height of image

    Notes
    -----
    Currently, this function only support rgb and grayscale which size of shape <= 3

    """

    shape = im.shape

    if len(shape) == 3:
        numChannel = shape[0]
        imHeight   = shape[1]
        imWidth    = shape[2]

    else:
        numChannel = 1
        imHeight   = shape[0]
        imWidth    = shape[1]


    return [numChannel, imWidth, imHeight]

def Resize(im,
           newSize = None,
           scaleFactor = None):
    """
    TODO: Need commment here

    Parameters
    ----------
    im

    Returns
    -------

    """
    if newSize is not None:
        newIm = cv2.resize(src   = im,
                           dsize = newSize)
    else:
        newIm = cv2.resize(src   = im,
                           dsize = (0, 0),
                           fx    = scaleFactor[0],
                           fy    = scaleFactor[1])

    return newIm

def ReadImage(imPath):
    CheckFileExist(imPath)

    return cv2.imread(imPath)

def ConvertImage(im, format):
    if format == 'gray':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def ScaleSize(oldSize, newSize):
    """
    Create new size from old size which is same aspect ratio with the old size

    Parameters
    ----------
    oldSize: tuple
        (width, heigh) - old size of image

    newSize: tuple
        (width, height) - new size of image

    Returns
    -------

    """

    if newSize[0] < 0 and newSize[1] < 0:
        return oldSize
    else:
        if newSize[0] < 0:
            ratio = newSize[1] * 1.0 / oldSize[1]
            newSize = (int(oldSize[0] * ratio), int(oldSize[1] * ratio))
            return newSize
        else:
            ratio = newSize[0] * 1.0 / oldSize[0]
            newSize = (int(oldSize[0] * ratio), int(oldSize[1] * ratio))
            return newSize

