import theano
import theano.tensor as T
import numpy
import cv2
from Layers.LayerHelper import *
from Layers.Net import *

class SSD512CustomModel():
    def __init__(self,
                 batchSize):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        X = T.tensor4('X')

        # Create shared variable for input
        net = ConvNeuralNet()
        net.NetName = 'SSD512CustomNet'
        net.NetOpts['batch_size'] = batchSize

        # Input
        net.Layer['input']                 = InputLayer(net, X)
        net.LayerOpts['reshape_new_shape'] = (net.NetOpts['batch_size'], 3, 512, 512)
        net.Layer['input_4d'] = ReshapeLayer(net, net.Layer['input'].Output)

        net.LayerOpts['pool_boder_mode']    = 1
        net.LayerOpts['conv2D_border_mode'] = 1

        # Stack 1
        net.LayerOpts['conv2D_filter_shape'] = (64, 3, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv1_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv1_1_b'
        net.Layer['conv1_1'] = ConvLayer(net, net.Layer['input_4d'].Output)
        net.Layer['relu1_1'] = ReLULayer(net.Layer['conv1_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (64, 64, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv1_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv1_2_b'
        net.Layer['conv1_2'] = ConvLayer(net, net.Layer['relu1_1'].Output)
        net.Layer['relu1_2'] = ReLULayer(net.Layer['conv1_2'].Output)

        net.LayerOpts['pool_mode'] = 'max'
        net.Layer['pool1']   = Pool2DLayer(net, net.Layer['relu1_2'].Output)

        # Stack 2
        net.LayerOpts['conv2D_filter_shape'] = (128, 64, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv2_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv2_1_b'
        net.Layer['conv2_1'] = ConvLayer(net, net.Layer['pool1'].Output)
        net.Layer['relu2_1'] = ReLULayer(net.Layer['conv2_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (128, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv2_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv2_2_b'
        net.Layer['conv2_2'] = ConvLayer(net, net.Layer['relu2_1'].Output)
        net.Layer['relu2_2'] = ReLULayer(net.Layer['conv2_2'].Output)

        net.Layer['pool2']   = Pool2DLayer(net, net.Layer['relu2_2'].Output)

        # Stack 3
        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv3_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv3_1_b'
        net.Layer['conv3_1'] = ConvLayer(net, net.Layer['pool2'].Output)
        net.Layer['relu3_1'] = ReLULayer(net.Layer['conv3_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv3_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv3_2_b'
        net.Layer['conv3_2'] = ConvLayer(net, net.Layer['relu3_1'].Output)
        net.Layer['relu3_2'] = ReLULayer(net.Layer['conv3_2'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv3_3_W'
        net.LayerOpts['conv2D_bName'] = 'conv3_3_b'
        net.Layer['conv3_3'] = ConvLayer(net, net.Layer['relu3_2'].Output)
        net.Layer['relu3_3'] = ReLULayer(net.Layer['conv3_3'].Output)

        net.Layer['pool3']   = Pool2DLayer(net, net.Layer['relu3_3'].Output)

        # Stack 4
        net.LayerOpts['conv2D_filter_shape'] = (512, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv4_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_1_b'
        net.Layer['conv4_1'] = ConvLayer(net, net.Layer['pool3'].Output)
        net.Layer['relu4_1'] = ReLULayer(net.Layer['conv4_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv4_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_2_b'
        net.Layer['conv4_2'] = ConvLayer(net, net.Layer['relu4_1'].Output)
        net.Layer['relu4_2'] = ReLULayer(net.Layer['conv4_2'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv4_3_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_b'
        net.Layer['conv4_3'] = ConvLayer(net, net.Layer['relu4_2'].Output)
        net.Layer['relu4_3'] = ReLULayer(net.Layer['conv4_3'].Output)

        net.Layer['pool4']   = Pool2DLayer(net, net.Layer['relu4_3'].Output)
        net.LayerOpts['normalize_scale']        = 20
        net.LayerOpts['normalize_filter_shape'] = (512, )
        net.LayerOpts['normalize_scale_name']   = 'conv4_3_scale'
        net.Layer['conv4_3_norm']        = NormalizeLayer(net, net.Layer['relu4_3'].Output)

        # conv4_3_norm_encode
        net.LayerOpts['conv2D_filter_shape'] = (256, 512, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv4_3_norm_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_norm_encode_b'
        net.Layer['conv4_3_norm_encode'] = ConvLayer(net, net.Layer['conv4_3_norm'].Output)
        net.Layer['conv4_3_norm_encode_relu'] = ReLULayer(net.Layer['conv4_3_norm_encode'].Output)

        # conv4_3_norm_decode
        W = net.Layer['conv4_3_norm_encode'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (512, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W']     = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'conv4_3_norm_decode_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_norm_decode_b'
        net.Layer['conv4_3_norm_decode'] = ConvLayer(net, net.Layer['conv4_3_norm_encode_relu'].Output)
        net.LayerOpts['conv2D_W']     = None   # Reset W for next layer

        # conv4_3_norm_encode_d1
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv4_3_norm_encode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_norm_encode_b_d1'
        net.Layer['conv4_3_norm_encode_d1']      = ConvLayer(net, net.Layer['conv4_3_norm_encode_relu'].Output)
        net.Layer['conv4_3_norm_encode_relu_d1'] = ReLULayer(net.Layer['conv4_3_norm_encode_d1'].Output)

        # conv4_3_norm_decode_d1
        W = net.Layer['conv4_3_norm_encode_d1'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (256, 1, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W']     = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'conv4_3_norm_decode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_norm_decode_b_d1'
        net.Layer['conv4_3_norm_decode_d1'] = ConvLayer(net, net.Layer['conv4_3_norm_encode_relu_d1'].Output)
        net.LayerOpts['conv2D_W'] = None  # Reset W for next layer

        # Stack 5
        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv5_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv5_1_b'
        net.Layer['conv5_1'] = ConvLayer(net, net.Layer['pool4'].Output)
        net.Layer['relu5_1'] = ReLULayer(net.Layer['conv5_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv5_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv5_2_b'
        net.Layer['conv5_2'] = ConvLayer(net, net.Layer['relu5_1'].Output)
        net.Layer['relu5_2'] = ReLULayer(net.Layer['conv5_2'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv5_3_W'
        net.LayerOpts['conv2D_bName'] = 'conv5_3_b'
        net.Layer['conv5_3'] = ConvLayer(net, net.Layer['relu5_2'].Output)
        net.Layer['relu5_3'] = ReLULayer(net.Layer['conv5_3'].Output)

        net.LayerOpts['pool_ignore_border'] = True
        net.LayerOpts['pool_filter_size']   = (3, 3)
        net.LayerOpts['pool_stride']        = (1, 1)
        net.LayerOpts['pool_padding']       = (1, 1)
        net.Layer['pool5']    = Pool2DLayer(net, net.Layer['relu5_3'].Output)

        # fc6 and fc7
        net.LayerOpts['conv2D_filter_shape']    = (1024, 512, 3, 3)
        net.LayerOpts['conv2D_stride']          = (1, 1)
        net.LayerOpts['conv2D_border_mode']     = (6, 6)
        net.LayerOpts['conv2D_filter_dilation'] = (6, 6)
        net.LayerOpts['conv2D_WName'] = 'fc6_W'
        net.LayerOpts['conv2D_bName'] = 'fc6_b'
        net.Layer['fc6']   = ConvLayer(net, net.Layer['pool5'].Output)
        net.Layer['relu6'] = ReLULayer(net.Layer['fc6'].Output)
        net.LayerOpts['conv2D_filter_dilation'] = (1, 1)        # Set default filter dilation

        net.LayerOpts['conv2D_filter_shape'] = (1024, 1024, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'fc7_W'
        net.LayerOpts['conv2D_bName']        = 'fc7_b'
        net.Layer['fc7']   = ConvLayer(net, net.Layer['relu6'].Output)
        net.Layer['relu7'] = ReLULayer(net.Layer['fc7'].Output)

        # First sub convolution to get predicted box
        # fc7_encode
        net.LayerOpts['conv2D_filter_shape'] = (256, 1024, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'fc7_encode_W'
        net.LayerOpts['conv2D_bName'] = 'fc7_encode_b'
        net.Layer['fc7_encode']      = ConvLayer(net, net.Layer['relu7'].Output)
        net.Layer['fc7_encode_relu'] = ReLULayer(net.Layer['fc7_encode'].Output)

        # fc7_decode
        W = net.Layer['fc7_encode'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (1024, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W']     = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'fc7_decode_W'
        net.LayerOpts['conv2D_bName'] = 'fc7_decode_b'
        net.Layer['fc7_decode'] = ConvLayer(net, net.Layer['fc7_encode_relu'].Output)
        net.LayerOpts['conv2D_W'] = None  # Reset W for next layer

        # fc7_encode_d1
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'fc7_encode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'fc7_encode_b_d1'
        net.Layer['fc7_encode_d1']      = ConvLayer(net, net.Layer['fc7_encode_relu'].Output)
        net.Layer['fc7_encode_relu_d1'] = ReLULayer(net.Layer['fc7_encode_d1'].Output)

        # fc7_decode_d1
        W = net.Layer['fc7_encode_d1'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (256, 1, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W'] = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'fc7_decode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'fc7_decode_b_d1'
        net.Layer['fc7_decode_d1'] = ConvLayer(net, net.Layer['fc7_encode_relu_d1'].Output)
        net.LayerOpts['conv2D_W']  = None  # Reset W for next layer

        # conv6_1 and conv6_2
        net.LayerOpts['conv2D_filter_shape'] = (256, 1024, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'conv6_1_W'
        net.LayerOpts['conv2D_bName']        = 'conv6_1_b'
        net.Layer['conv6_1']      = ConvLayer(net, net.Layer['relu7'].Output)
        net.Layer['conv6_1_relu'] = ReLULayer(net.Layer['conv6_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'conv6_2_W'
        net.LayerOpts['conv2D_bName']        = 'conv6_2_b'
        net.Layer['conv6_2'] = ConvLayer(net, net.Layer['conv6_1_relu'].Output)
        net.Layer['conv6_2_relu'] = ReLULayer(net.Layer['conv6_2'].Output)

        # Second sub convolution to get predicted box
        # conv6_2_encode
        net.LayerOpts['conv2D_filter_shape'] = (256, 512, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv6_2_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv6_2_encode_b'
        net.Layer['conv6_2_encode'] = ConvLayer(net, net.Layer['conv6_2_relu'].Output)
        net.Layer['conv6_2_encode_relu'] = ReLULayer(net.Layer['conv6_2_encode'].Output)

        # conv6_2_decode
        W = net.Layer['conv6_2_encode'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (512, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W']            = WLayer.Output
        net.LayerOpts['conv2D_WName']        = 'conv6_2_decode_W'
        net.LayerOpts['conv2D_bName']        = 'conv6_2_decode_b'
        net.Layer['conv6_2_decode'] = ConvLayer(net, net.Layer['conv6_2_encode_relu'].Output)
        net.LayerOpts['conv2D_W']   = None  # Reset W for next layer

        # conv6_2_encode_d1
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv6_2_encode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv6_2_encode_b_d1'
        net.Layer['conv6_2_encode_d1']      = ConvLayer(net, net.Layer['conv6_2_encode_relu'].Output)
        net.Layer['conv6_2_encode_relu_d1'] = ReLULayer(net.Layer['conv6_2_encode_d1'].Output)

        # conv6_2_decode_d1
        W = net.Layer['conv6_2_encode_d1'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (256, 1, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W'] = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'conv6_2_decode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv6_2_decode_b_d1'
        net.Layer['conv6_2_decode_d1'] = ConvLayer(net, net.Layer['conv6_2_encode_relu_d1'].Output)
        net.LayerOpts['conv2D_W'] = None  # Reset W for next layer

        # conv7_1 and conv7_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 512, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'conv7_1_W'
        net.LayerOpts['conv2D_bName']        = 'conv7_1_b'
        net.Layer['conv7_1']      = ConvLayer(net, net.Layer['conv6_2_relu'].Output)
        net.Layer['conv7_1_relu'] = ReLULayer(net.Layer['conv7_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'conv7_2_W'
        net.LayerOpts['conv2D_bName']        = 'conv7_2_b'
        net.Layer['conv7_2']      = ConvLayer(net, net.Layer['conv7_1_relu'].Output)
        net.Layer['conv7_2_relu'] = ReLULayer(net.Layer['conv7_2'].Output)

        # conv7_2_encode_d1
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv7_2_encode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv7_2_encode_b_d1'
        net.Layer['conv7_2_encode_d1']      = ConvLayer(net, net.Layer['conv7_2_relu'].Output)
        net.Layer['conv7_2_encode_relu_d1'] = ReLULayer(net.Layer['conv7_2_encode_d1'].Output)

        # conv7_2_decode_d1
        W = net.Layer['conv7_2_encode_d1'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (256, 1, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W'] = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'conv7_2_decode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv7_2_decode_b_d1'
        net.Layer['conv7_2_decode_d1'] = ConvLayer(net, net.Layer['conv7_2_encode_relu_d1'].Output)
        net.LayerOpts['conv2D_W'] = None  # Reset W for next layer

        # Third sub convolution to get predicted box
        # conv8_1 and conv8_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'conv8_1_W'
        net.LayerOpts['conv2D_bName']        = 'conv8_1_b'
        net.Layer['conv8_1']      = ConvLayer(net, net.Layer['conv7_2_relu'].Output)
        net.Layer['conv8_1_relu'] = ReLULayer(net.Layer['conv8_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'conv8_2_W'
        net.LayerOpts['conv2D_bName']        = 'conv8_2_b'
        net.Layer['conv8_2'] = ConvLayer(net, net.Layer['conv8_1_relu'].Output)
        net.Layer['conv8_2_relu'] = ReLULayer(net.Layer['conv8_2'].Output)

        # conv8_2_encode_d1
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv8_2_encode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv8_2_encode_b_d1'
        net.Layer['conv8_2_encode_d1']      = ConvLayer(net, net.Layer['conv8_2_relu'].Output)
        net.Layer['conv8_2_encode_relu_d1'] = ReLULayer(net.Layer['conv8_2_encode_d1'].Output)

        # conv8_2_decode_d1
        W = net.Layer['conv8_2_encode_d1'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (256, 1, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W'] = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'conv8_2_decode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv8_2_decode_b_d1'
        net.Layer['conv8_2_decode_d1'] = ConvLayer(net, net.Layer['conv8_2_encode_relu_d1'].Output)
        net.LayerOpts['conv2D_W'] = None  # Reset W for next layer

        # Fourth sub convolution to get predicted box
        # conv9_1 and conv9_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'conv9_1_W'
        net.LayerOpts['conv2D_bName']        = 'conv9_1_b'
        net.Layer['conv9_1']      = ConvLayer(net, net.Layer['conv8_2_relu'].Output)
        net.Layer['conv9_1_relu'] = ReLULayer(net.Layer['conv9_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv9_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_b'
        net.Layer['conv9_2']      = ConvLayer(net, net.Layer['conv9_1_relu'].Output)
        net.Layer['conv9_2_relu'] = ReLULayer(net.Layer['conv9_2'].Output)

        # conv9_2_encode_d1
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv9_2_encode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_encode_b_d1'
        net.Layer['conv9_2_encode_d1'] = ConvLayer(net, net.Layer['conv9_2_relu'].Output)
        net.Layer['conv9_2_encode_relu_d1'] = ReLULayer(net.Layer['conv9_2_encode_d1'].Output)

        # conv9_2_decode_d1
        W = net.Layer['conv9_2_encode_d1'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (256, 1, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W'] = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'conv9_2_decode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_decode_b_d1'
        net.Layer['conv9_2_decode_d1'] = ConvLayer(net, net.Layer['conv9_2_encode_relu_d1'].Output)
        net.LayerOpts['conv2D_W'] = None  # Reset W for next layer

        # Fifth sub convolution to get predicted box
        # conv10_1 and conv10_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName'] = 'conv10_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv10_1_b'
        net.Layer['conv10_1'] = ConvLayer(net, net.Layer['conv9_2_relu'].Output)
        net.Layer['conv10_1_relu'] = ReLULayer(net.Layer['conv10_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 4, 4)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv10_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv10_2_b'
        net.Layer['conv10_2'] = ConvLayer(net, net.Layer['conv10_1_relu'].Output)
        net.Layer['conv10_2_relu'] = ReLULayer(net.Layer['conv10_2'].Output)

        # conv10_2_encode_d1
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_WName'] = 'conv10_2_encode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv10_2_encode_b_d1'
        net.Layer['conv10_2_encode_d1']      = ConvLayer(net, net.Layer['conv10_2_relu'].Output)
        net.Layer['conv10_2_encode_relu_d1'] = ReLULayer(net.Layer['conv10_2_encode_d1'].Output)

        # conv10_2_decode_d1
        W = net.Layer['conv10_2_encode_d1'].W
        net.LayerOpts['permute_dimension'] = (1, 0, 2, 3)
        WLayer = PermuteLayer(net, W)
        net.LayerOpts['conv2D_filter_shape'] = (256, 1, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (0, 0)
        net.LayerOpts['conv2D_W'] = WLayer.Output
        net.LayerOpts['conv2D_WName'] = 'conv10_2_decode_W_d1'
        net.LayerOpts['conv2D_bName'] = 'conv10_2_decode_b_d1'
        net.Layer['conv10_2_decode_d1'] = ConvLayer(net, net.Layer['conv10_2_encode_relu_d1'].Output)
        net.LayerOpts['conv2D_W'] = None  # Reset W for next layer

        self.Net = net

        # Train conv4_3_encode layer
        net.LayerOpts['flatten_ndim']  = 2
        net.Layer['conv4_3_norm_flat'] = FlattenLayer(net, net.Layer['conv4_3_norm'].Output)
        net.LayerOpts['flatten_ndim']         = 2
        net.Layer['conv4_3_norm_decode_flat'] = FlattenLayer(net, net.Layer['conv4_3_norm_decode'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv4_3_encode_costl2 = L2CostLayer(net, net.Layer['conv4_3_norm_flat'].Output,
                                            net.Layer['conv4_3_norm_decode_flat'].Output).Output
        params  = net.Layer['conv4_3_norm_encode'].Params
        grads   = T.grad(conv4_3_encode_costl2, params)
        updates = AdamGDUpdate(net, params = params, grads = grads).Updates
        self.Conv4_3EncodeFunc = theano.function(
                                    inputs  = [X],
                                    updates = updates,
                                    outputs = conv4_3_encode_costl2)

        # Train conv4_3_encode_d1 layer
        net.LayerOpts['flatten_ndim']              = 2
        net.Layer['conv4_3_norm_encode_relu_flat'] = FlattenLayer(net, net.Layer['conv4_3_norm_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']            = 2
        net.Layer['conv4_3_norm_decode_d1_flat'] = FlattenLayer(net, net.Layer['conv4_3_norm_decode_d1'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv4_3_encode_d1_costl2 = L2CostLayer(net, net.Layer['conv4_3_norm_encode_relu_flat'].Output,
                                               net.Layer['conv4_3_norm_decode_d1_flat'].Output).Output
        params = net.Layer['conv4_3_norm_encode_d1'].Params
        grads  = T.grad(conv4_3_encode_d1_costl2, params)
        updates = AdamGDUpdate(net, params = params, grads = grads).Updates
        self.Conv4_3EncodeD1Func = theano.function(
                                    inputs  = [X],
                                    updates = updates,
                                    outputs = conv4_3_encode_d1_costl2)

        # Train fc7_encode layer
        net.LayerOpts['flatten_ndim']  = 2
        net.Layer['relu7_flat']        = FlattenLayer(net, net.Layer['relu7'].Output)
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['fc7_decode_flat']  = FlattenLayer(net, net.Layer['fc7_decode'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        fc7_encode_costl2            = L2CostLayer(net, net.Layer['relu7_flat'].Output,
                                                   net.Layer['fc7_decode_flat'].Output).Output
        params  = net.Layer['fc7_encode'].Params
        grads   = T.grad(fc7_encode_costl2, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates
        self.Fc7EncodeFunc = theano.function(
                                    inputs  = [X],
                                    updates = updates,
                                    outputs = fc7_encode_costl2)

        # Train fc7_encode_d1 layer
        net.LayerOpts['flatten_ndim']       = 2
        net.Layer['relu7_encode_relu_flat'] = FlattenLayer(net, net.Layer['fc7_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']   = 2
        net.Layer['fc7_decode_d1_flat'] = FlattenLayer(net, net.Layer['fc7_decode_d1'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        fc7_encode_d1_costl2 = L2CostLayer(net, net.Layer['relu7_encode_relu_flat'].Output,
                                           net.Layer['fc7_decode_d1_flat'].Output).Output
        params = net.Layer['fc7_encode_d1'].Params
        grads  = T.grad(fc7_encode_d1_costl2, params)
        updates = AdamGDUpdate(net, params = params, grads = grads).Updates
        self.Fc7EncodeD1Func = theano.function(
                                    inputs  = [X],
                                    updates = updates,
                                    outputs = fc7_encode_d1_costl2)

        # Train conv6_2_encode layer
        net.LayerOpts['flatten_ndim']  = 2
        net.Layer['conv6_2_relu_flat'] = FlattenLayer(net, net.Layer['conv6_2_relu'].Output)
        net.LayerOpts['flatten_ndim']    = 2
        net.Layer['conv6_2_decode_flat'] = FlattenLayer(net, net.Layer['conv6_2_decode'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv6_2_encode_costl2        = L2CostLayer(net, net.Layer['conv6_2_relu_flat'].Output,
                                                   net.Layer['conv6_2_decode_flat'].Output).Output
        params = net.Layer['conv6_2_encode'].Params
        grads  = T.grad(conv6_2_encode_costl2, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates
        self.Conv6_2EncodeFunc = theano.function(
                                    inputs  = [X],
                                    updates = updates,
                                    outputs = conv6_2_encode_costl2)

        # Train conv6_2_encode_d1 layer
        net.LayerOpts['flatten_ndim']         = 2
        net.Layer['conv6_2_encode_relu_flat'] = FlattenLayer(net, net.Layer['conv6_2_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']       = 2
        net.Layer['conv6_2_decode_d1_flat'] = FlattenLayer(net, net.Layer['conv6_2_decode_d1'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv6_2_encode_d1_costl2 = L2CostLayer(net, net.Layer['conv6_2_encode_relu_flat'].Output,
                                               net.Layer['conv6_2_decode_d1_flat'].Output).Output
        params = net.Layer['conv6_2_encode_d1'].Params
        grads = T.grad(conv6_2_encode_d1_costl2, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates
        self.Conv6_2EncodeD1Func = theano.function(
                                    inputs  = [X],
                                    updates = updates,
                                    outputs = conv6_2_encode_d1_costl2)

        # Train conv7_2_encode_d1 layer
        net.LayerOpts['flatten_ndim']  = 2
        net.Layer['conv7_2_relu_flat'] = FlattenLayer(net, net.Layer['conv7_2_relu'].Output)
        net.LayerOpts['flatten_ndim']       = 2
        net.Layer['conv7_2_decode_d1_flat'] = FlattenLayer(net, net.Layer['conv7_2_decode_d1'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv7_2_encode_d1_costl2 = L2CostLayer(net, net.Layer['conv7_2_relu_flat'].Output,
                                               net.Layer['conv7_2_decode_d1_flat'].Output).Output
        params = net.Layer['conv7_2_encode_d1'].Params
        grads = T.grad(conv7_2_encode_d1_costl2, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates
        self.Conv7_2EncodeD1Func = theano.function(
                                        inputs  = [X],
                                        updates = updates,
                                        outputs = conv7_2_encode_d1_costl2)

        # Train conv8_2_encode_d1 layer
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['conv8_2_relu_flat'] = FlattenLayer(net, net.Layer['conv8_2_relu'].Output)
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['conv8_2_decode_d1_flat'] = FlattenLayer(net, net.Layer['conv8_2_decode_d1'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv8_2_encode_d1_costl2 = L2CostLayer(net, net.Layer['conv8_2_relu_flat'].Output,
                                               net.Layer['conv8_2_decode_d1_flat'].Output).Output
        params = net.Layer['conv8_2_encode_d1'].Params
        grads = T.grad(conv8_2_encode_d1_costl2, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates
        self.Conv8_2EncodeD1Func = theano.function(
                                        inputs=[X],
                                        updates=updates,
                                        outputs=conv8_2_encode_d1_costl2)

        # Train conv9_2_encode_d1 layer
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['conv9_2_relu_flat'] = FlattenLayer(net, net.Layer['conv9_2_relu'].Output)
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['conv9_2_decode_d1_flat'] = FlattenLayer(net, net.Layer['conv9_2_decode_d1'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv9_2_encode_d1_costl2 = L2CostLayer(net, net.Layer['conv9_2_relu_flat'].Output,
                                               net.Layer['conv9_2_decode_d1_flat'].Output).Output
        params = net.Layer['conv9_2_encode_d1'].Params
        grads = T.grad(conv9_2_encode_d1_costl2, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates
        self.Conv9_2EncodeD1Func = theano.function(
                                        inputs=[X],
                                        updates=updates,
                                        outputs=conv9_2_encode_d1_costl2)

        # Train conv10_2_encode_d1 layer
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['conv10_2_relu_flat'] = FlattenLayer(net, net.Layer['conv10_2_relu'].Output)
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['conv10_2_decode_d1_flat'] = FlattenLayer(net, net.Layer['conv10_2_decode_d1'].Output)
        net.LayerOpts['l2cost_axis'] = 1
        conv10_2_encode_d1_costl2 = L2CostLayer(net, net.Layer['conv10_2_relu_flat'].Output,
                                               net.Layer['conv10_2_decode_d1_flat'].Output).Output
        params = net.Layer['conv10_2_encode_d1'].Params
        grads = T.grad(conv10_2_encode_d1_costl2, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates
        self.Conv10_2EncodeD1Func = theano.function(
                                        inputs=[X],
                                        updates=updates,
                                        outputs=conv10_2_encode_d1_costl2)

    def LoadCaffeModel(self,
                      caffePrototxtPath,
                      caffeModelPath):
        self.Net.LoadCaffeModel(caffePrototxtPath, caffeModelPath)

    def SaveLayers(self,
                   file,
                   layerNames):
        for layerName in layerNames:
            self.Net.Layer[layerName].SaveModel(file = file)

    def LoadLayers(self,
                   file,
                   layerNames):
        for layerName in layerNames:
            self.Net.Layer[layerName].LoadModel(file = file)

    def TestNetwork(self,
                    im):
        return self.PredFunc(im)
