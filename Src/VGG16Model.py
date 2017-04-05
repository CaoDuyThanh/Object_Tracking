import theano
import theano.tensor as T
import numpy
import cv2
from Layers.LayerHelper import *
from Layers.Net import *

BATCH_SIZE = 1

class VGG16Model():
    def __init__(self):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        X = T.tensor4('X')
        Y = T.ivector('Y')

        # Create shared variable for input
        net = ConvNeuralNet()
        net.NetName = 'VGG16Net'

        # Input
        net.Layer['input'] = InputLayer(net, X)
        net.LayerOpts['reshape_new_shape'] = (net.NetOpts['batch_size'], 3, 224, 224)
        net.Layer['input_4d'] = ReshapeLayer(net, net.Layer['input'].Output)

        net.LayerOpts['pool_boder_mode'] = 1
        net.LayerOpts['conv2D_border_mode'] = 1

        # Stack 1
        net.LayerOpts['conv2D_filter_shape'] = (64, 3, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv1_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv1_1_b'
        net.Layer['conv1_1'] = ConvLayer(net, net.Layer['input_4d'].Output)
        net.Layer['relu1_1'] = ReLULayer(net.Layer['conv1_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (64, 64, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv1_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv1_2_b'
        net.Layer['conv1_2'] = ConvLayer(net, net.Layer['relu1_1'].Output)
        net.Layer['relu1_2'] = ReLULayer(net.Layer['conv1_2'].Output)

        net.LayerOpts['pool_mode'] = 'max'
        net.Layer['pool1']   = Pool2DLayer(net, net.Layer['relu1_2'].Output)

        # Stack 2
        net.LayerOpts['conv2D_filter_shape'] = (128, 64, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv2_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv2_1_b'
        net.Layer['conv2_1'] = ConvLayer(net, net.Layer['pool1'].Output)
        net.Layer['relu2_1'] = ReLULayer(net.Layer['conv2_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (128, 128, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv2_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv2_2_b'
        net.Layer['conv2_2'] = ConvLayer(net, net.Layer['relu2_1'].Output)
        net.Layer['relu2_2'] = ReLULayer(net.Layer['conv2_2'].Output)

        net.Layer['pool2']   = Pool2DLayer(net, net.Layer['relu2_2'].Output)

        # Stack 3
        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv3_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv3_1_b'
        net.Layer['conv3_1'] = ConvLayer(net, net.Layer['pool2'].Output)
        net.Layer['relu3_1'] = ReLULayer(net.Layer['conv3_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 256, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv3_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv3_2_b'
        net.Layer['conv3_2'] = ConvLayer(net, net.Layer['relu3_1'].Output)
        net.Layer['relu3_2'] = ReLULayer(net.Layer['conv3_2'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 256, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv3_3_W'
        net.LayerOpts['conv2D_bName'] = 'conv3_3_b'
        net.Layer['conv3_3'] = ConvLayer(net, net.Layer['relu3_2'].Output)
        net.Layer['relu3_3'] = ReLULayer(net.Layer['conv3_3'].Output)

        net.Layer['pool3']   = Pool2DLayer(net, net.Layer['relu3_3'].Output)

        # Stack 4
        net.LayerOpts['conv2D_filter_shape'] = (512, 256, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv4_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_1_b'
        net.Layer['conv4_1'] = ConvLayer(net, net.Layer['pool3'].Output)
        net.Layer['relu4_1'] = ReLULayer(net.Layer['conv4_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv4_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_2_b'
        net.Layer['conv4_2'] = ConvLayer(net, net.Layer['relu4_1'].Output)
        net.Layer['relu4_2'] = ReLULayer(net.Layer['conv4_2'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv4_3_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_b'
        net.Layer['conv4_3'] = ConvLayer(net, net.Layer['relu4_2'].Output)
        net.Layer['relu4_3'] = ReLULayer(net.Layer['conv4_3'].Output)

        net.Layer['pool4']   = Pool2DLayer(net, net.Layer['relu4_3'].Output)

        # Stack 5
        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv5_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv5_1_b'
        net.Layer['conv5_1'] = ConvLayer(net, net.Layer['pool4'].Output)
        net.Layer['relu5_1'] = ReLULayer(net.Layer['conv5_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv5_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv5_2_b'
        net.Layer['conv5_2'] = ConvLayer(net, net.Layer['relu5_1'].Output)
        net.Layer['relu5_2'] = ReLULayer(net.Layer['conv5_2'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_WName'] = 'conv5_3_W'
        net.LayerOpts['conv2D_bName'] = 'conv5_3_b'
        net.Layer['conv5_3'] = ConvLayer(net, net.Layer['relu5_2'].Output)
        net.Layer['relu5_3'] = ReLULayer(net.Layer['conv5_3'].Output)

        net.Layer['pool5']    = Pool2DLayer(net, net.Layer['relu5_3'].Output)

        # Reshape pool5 to 2d
        net.LayerOpts['reshape_new_shape'] = (net.NetOpts['batch_size'], 512 * 7 * 7)
        net.Layer['pool5_2d'] = ReshapeLayer(net, net.Layer['pool5'].Output)

        # fc6
        net.LayerOpts['hidden_input_size']  = 512 * 7 * 7
        net.LayerOpts['hidden_output_size'] = 4096
        net.LayerOpts['hidden_WName'] = 'fc6_W'
        net.LayerOpts['hidden_bName'] = 'fc6_b'
        net.Layer['fc6']     = HiddenLayer(net, net.Layer['pool5_2d'].Output)
        net.Layer['relu6']   = ReLULayer(net.Layer['fc6'].Output)
        net.Layer['drop6']   = DropoutLayer(net, net.Layer['relu6'].Output)

        # fc7
        net.LayerOpts['hidden_input_size']  = 4096
        net.LayerOpts['hidden_output_size'] = 4096
        net.LayerOpts['hidden_WName'] = 'fc7_W'
        net.LayerOpts['hidden_bName'] = 'fc7_b'
        net.Layer['fc7']     = HiddenLayer(net, net.Layer['drop6'].Output)
        net.Layer['relu7']   = ReLULayer(net.Layer['fc7'].Output)
        net.Layer['drop7']   = DropoutLayer(net, net.Layer['relu7'].Output)

        # fc8
        net.LayerOpts['hidden_input_size']  = 4096
        net.LayerOpts['hidden_output_size'] = 1000
        net.LayerOpts['hidden_WName'] = 'fc8_W'
        net.LayerOpts['hidden_bName'] = 'fc8_b'
        net.Layer['fc8']     = HiddenLayer(net, net.Layer['drop7'].Output)
        net.Layer['prob']    = SoftmaxLayer(net.Layer['fc8'].Output)

        self.Net = net

        # Predict function
        pred = T.argmax(net.Layer['prob'].Output, axis = 1)
        self.PredFunc = theano.function(
                            inputs  = [X],
                            outputs = pred)

        self.ProbFunc = theano.function(
                            inputs  = [X],
                            outputs = net.Layer['prob'].Output)

    def LoadCaffeModel(self,
                  caffePrototxtPath,
                  caffeModelPath):
        self.Net.LoadCaffeModel(caffePrototxtPath, caffeModelPath)

    def TestNetwork(self,
                    im):
        return self.ProbFunc(im)
