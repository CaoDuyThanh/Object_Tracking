import theano
import theano.tensor as T
import math
import numpy
import cv2
import matplotlib.pyplot as plt
import skimage.transform
from Layers.LayerHelper import *

class SSD512FeaExtraction():
    def __init__(self):
        ####################################
        #       Create model               #
        ####################################
        # Create tensor variables to store input / output data
        self.X = T.tensor4('X')

        # Create shared variable for input
        net = ConvNeuralNet()
        net.NetName = 'SSD512CustomNet'

        # Input
        net.Layer['input_4d'] = InputLayer(net, self.X)
        # net.LayerOpts['reshape_new_shape'] = (net.NetOpts['batch_size'], 3, 512, 512)
        # net.Layer['input_4d'] = ReshapeLayer(net, net.Layer['input'].Output)

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
        net.Layer['pool1'] = Pool2DLayer(net, net.Layer['relu1_2'].Output)

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

        net.Layer['pool2'] = Pool2DLayer(net, net.Layer['relu2_2'].Output)

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

        net.Layer['pool3'] = Pool2DLayer(net, net.Layer['relu3_3'].Output)

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

        net.Layer['pool4'] = Pool2DLayer(net, net.Layer['relu4_3'].Output)
        net.LayerOpts['normalize_scale']        = 20
        net.LayerOpts['normalize_filter_shape'] = (512,)
        net.LayerOpts['normalize_scale_name']   = 'conv4_3_scale'
        net.Layer['conv4_3_norm'] = NormalizeLayer(net, net.Layer['relu4_3'].Output)

        # conv4_3_norm_encode
        net.LayerOpts['conv2D_filter_shape'] = (1, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv4_3_norm_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_norm_encode_b'
        net.Layer['conv4_3_norm_encode']      = ConvLayer(net, net.Layer['conv4_3_norm'].Output)
        net.Layer['conv4_3_norm_encode_relu'] = ReLULayer(net.Layer['conv4_3_norm_encode'].Output)

        # conv4_3_norm_flat
        net.LayerOpts['permute_dimension']    = (0, 2, 3, 1)
        net.Layer['conv4_3_norm_encode_perm'] = PermuteLayer(net, net.Layer['conv4_3_norm_encode_relu'].Output)
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['conv4_3_norm_encode_flat'] = FlattenLayer(net, net.Layer['conv4_3_norm_encode_perm'].Output)

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
        net.LayerOpts['pool_stride']  = (1, 1)
        net.LayerOpts['pool_padding'] = (1, 1)
        net.Layer['pool5'] = Pool2DLayer(net, net.Layer['relu5_3'].Output)

        # fc6 and fc7
        net.LayerOpts['conv2D_filter_shape'] = (1024, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (6, 6)
        net.LayerOpts['conv2D_filter_dilation'] = (6, 6)
        net.LayerOpts['conv2D_WName'] = 'fc6_W'
        net.LayerOpts['conv2D_bName'] = 'fc6_b'
        net.Layer['fc6']   = ConvLayer(net, net.Layer['pool5'].Output)
        net.Layer['relu6'] = ReLULayer(net.Layer['fc6'].Output)
        net.LayerOpts['conv2D_filter_dilation'] = (1, 1)  # Set default filter dilation

        net.LayerOpts['conv2D_filter_shape'] = (1024, 1024, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName'] = 'fc7_W'
        net.LayerOpts['conv2D_bName'] = 'fc7_b'
        net.Layer['fc7'] = ConvLayer(net, net.Layer['relu6'].Output)
        net.Layer['relu7'] = ReLULayer(net.Layer['fc7'].Output)

        # fc7_encode
        net.LayerOpts['conv2D_filter_shape'] = (1, 1024, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'fc7_encode_W'
        net.LayerOpts['conv2D_bName'] = 'fc7_encode_b'
        net.Layer['fc7_encode']      = ConvLayer(net, net.Layer['relu7'].Output)
        net.Layer['fc7_encode_relu'] = ReLULayer(net.Layer['fc7_encode'].Output)

        # fc7_encode_flat
        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['fc7_encode_perm']       = PermuteLayer(net, net.Layer['fc7_encode_relu'].Output)
        net.LayerOpts['flatten_ndim'] = 2
        net.Layer['fc7_encode_flat']  = FlattenLayer(net, net.Layer['fc7_encode_perm'].Output)

        # First sub convolution to get predicted box
        # conv6_1 and conv6_2
        net.LayerOpts['conv2D_filter_shape'] = (256, 1024, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName'] = 'conv6_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv6_1_b'
        net.Layer['conv6_1'] = ConvLayer(net, net.Layer['relu7'].Output)
        net.Layer['conv6_1_relu'] = ReLULayer(net.Layer['conv6_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (512, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv6_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv6_2_b'
        net.Layer['conv6_2'] = ConvLayer(net, net.Layer['conv6_1_relu'].Output)
        net.Layer['conv6_2_relu'] = ReLULayer(net.Layer['conv6_2'].Output)

        # Second sub convolution to get predicted box
        # conv6_2_encode
        net.LayerOpts['conv2D_filter_shape'] = (1, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv6_2_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv6_2_encode_b'
        net.Layer['conv6_2_encode']      = ConvLayer(net, net.Layer['conv6_2_relu'].Output)
        net.Layer['conv6_2_encode_relu'] = ReLULayer(net.Layer['conv6_2_encode'].Output)

        # conv6_2_encode_flat
        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv6_2_encode_perm']   = PermuteLayer(net, net.Layer['conv6_2_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']    = 2
        net.Layer['conv6_2_encode_flat'] = FlattenLayer(net, net.Layer['conv6_2_encode_perm'].Output)

        # conv7_1 and conv7_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 512, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName'] = 'conv7_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv7_1_b'
        net.Layer['conv7_1']      = ConvLayer(net, net.Layer['conv6_2_relu'].Output)
        net.Layer['conv7_1_relu'] = ReLULayer(net.Layer['conv7_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv7_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv7_2_b'
        net.Layer['conv7_2']      = ConvLayer(net, net.Layer['conv7_1_relu'].Output)
        net.Layer['conv7_2_relu'] = ReLULayer(net.Layer['conv7_2'].Output)

        # conv7_2_encode
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv7_2_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv7_2_encode_b'
        net.Layer['conv7_2_encode']      = ConvLayer(net, net.Layer['conv7_2_relu'].Output)
        net.Layer['conv7_2_encode_relu'] = ReLULayer(net.Layer['conv7_2_encode'].Output)

        # conv7_2_encode_flat
        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv7_2_encode_perm']   = PermuteLayer(net, net.Layer['conv7_2_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']    = 2
        net.Layer['conv7_2_encode_flat'] = FlattenLayer(net, net.Layer['conv7_2_encode_perm'].Output)

        # Third sub convolution to get predicted box
        # conv8_1 and conv8_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName'] = 'conv8_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv8_1_b'
        net.Layer['conv8_1']      = ConvLayer(net, net.Layer['conv7_2_relu'].Output)
        net.Layer['conv8_1_relu'] = ReLULayer(net.Layer['conv8_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv8_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv8_2_b'
        net.Layer['conv8_2']      = ConvLayer(net, net.Layer['conv8_1_relu'].Output)
        net.Layer['conv8_2_relu'] = ReLULayer(net.Layer['conv8_2'].Output)

        # conv8_2_encode
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv8_2_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv8_2_encode_b'
        net.Layer['conv8_2_encode']      = ConvLayer(net, net.Layer['conv8_2_relu'].Output)
        net.Layer['conv8_2_encode_relu'] = ReLULayer(net.Layer['conv8_2_encode'].Output)

        # conv8_2_encode_flat
        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv8_2_encode_perm']   = PermuteLayer(net, net.Layer['conv8_2_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']    = 2
        net.Layer['conv8_2_encode_flat'] = FlattenLayer(net, net.Layer['conv8_2_encode_perm'].Output)

        # Fourth sub convolution to get predicted box
        # conv9_1 and conv9_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']      = (1, 1)
        net.LayerOpts['conv2D_border_mode'] = 0
        net.LayerOpts['conv2D_WName']       = 'conv9_1_W'
        net.LayerOpts['conv2D_bName']       = 'conv9_1_b'
        net.Layer['conv9_1']      = ConvLayer(net, net.Layer['conv8_2_relu'].Output)
        net.Layer['conv9_1_relu'] = ReLULayer(net.Layer['conv9_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (2, 2)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv9_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_b'
        net.Layer['conv9_2']      = ConvLayer(net, net.Layer['conv9_1_relu'].Output)
        net.Layer['conv9_2_relu'] = ReLULayer(net.Layer['conv9_2'].Output)

        # conv9_2_encode
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv9_2_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_encode_b'
        net.Layer['conv9_2_encode']      = ConvLayer(net, net.Layer['conv9_2_relu'].Output)
        net.Layer['conv9_2_encode_relu'] = ReLULayer(net.Layer['conv9_2_encode'].Output)

        # conv9_2_encode_flat
        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv9_2_encode_perm']   = PermuteLayer(net, net.Layer['conv9_2_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']    = 2
        net.Layer['conv9_2_encode_flat'] = FlattenLayer(net, net.Layer['conv9_2_encode_perm'].Output)

        # Fourth sub convolution to get predicted box
        # conv10_1 and conv10_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName'] = 'conv10_1_W'
        net.LayerOpts['conv2D_bName'] = 'conv10_1_b'
        net.Layer['conv10_1']      = ConvLayer(net, net.Layer['conv9_2_relu'].Output)
        net.Layer['conv10_1_relu'] = ReLULayer(net.Layer['conv10_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 4, 4)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv10_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv10_2_b'
        net.Layer['conv10_2']      = ConvLayer(net, net.Layer['conv10_1_relu'].Output)
        net.Layer['conv10_2_relu'] = ReLULayer(net.Layer['conv10_2'].Output)

        # conv10_2_encode
        net.LayerOpts['conv2D_filter_shape'] = (1, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv10_2_encode_W'
        net.LayerOpts['conv2D_bName'] = 'conv10_2_encode_b'
        net.Layer['conv10_2_encode']      = ConvLayer(net, net.Layer['conv10_2_relu'].Output)
        net.Layer['conv10_2_encode_relu'] = ReLULayer(net.Layer['conv10_2_encode'].Output)

        # conv10_2_encode_flat
        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv10_2_encode_perm']  = PermuteLayer(net, net.Layer['conv10_2_encode_relu'].Output)
        net.LayerOpts['flatten_ndim']     = 2
        net.Layer['conv10_2_encode_flat'] = FlattenLayer(net, net.Layer['conv10_2_encode_perm'].Output)

        # Concat features
        net.Layer['features'] = ConcatLayer(net, [net.Layer['conv4_3_norm_encode_flat'].Output,
                                                  net.Layer['fc7_encode_flat'].Output,
                                                  net.Layer['conv6_2_encode_flat'].Output,
                                                  net.Layer['conv7_2_encode_flat'].Output,
                                                  net.Layer['conv8_2_encode_flat'].Output,
                                                  net.Layer['conv9_2_encode_flat'].Output,
                                                  net.Layer['conv10_2_encode_flat'].Output])

        net.LayerOpts['reshape_new_shape'] = (net.Layer['input_4d'].Output.shape[0], 5461, 1)
        net.Layer['features_reshape']      = ReshapeLayer(net, net.Layer['features'].Output)

        self.Net = net

        # self.FeatureFunc = theano.function(inputs  = [X],
        #                                    outputs = [net.Layer['features_reshape'].Output])

    def GetDefaultBbox(self,
                       imageWidth,
                       sMin       = 10,
                       sMax       = 90,
                       layerSizes = [],
                       numBoxs    = [],
                       offset     = 0.5,
                       steps      = []):
        minSizes = []
        maxSizes = []
        step     = int(math.floor((sMax - sMin) / (len(layerSizes) - 2)))
        for ratio in xrange(sMin, sMax + 1, step):
            minSizes.append(imageWidth *  ratio         / 100.)
            maxSizes.append(imageWidth * (ratio + step) / 100.)
        minSizes = [imageWidth *  4 / 100.] + minSizes
        maxSizes = [imageWidth * 10 / 100.] + maxSizes

        defaultBboxs = []
        for k, layerSize in enumerate(layerSizes):
            layerWidth  = layerSize[0]
            layerHeight = layerSize[1]
            numbox      = numBoxs[k]

            minSize = minSizes[k]
            maxSize = maxSizes[k]

            if numbox == 4:
                aspectRatio = [1., 2., 1. / 2.]
            elif numbox == 6:
                aspectRatio = [1., 2., 1. / 2., 3., 1. / 3.]
            elif numbox == 8:
                aspectRatio = [1., 2., 1. / 2., 3., 1. / 3., 6. / 1., 1. / 6.]
            stepW = stepH = steps[k]
            for h in range(layerHeight):
                for w in range(layerWidth):
                    centerX = (w + offset) * stepW
                    centerY = (h + offset) * stepH

                    defaultBbox = []
                    # first prior: aspect_ratio = 1, size = min_size
                    boxWidth = boxHeight = minSize
                    # cx | cy
                    cx = centerX / imageWidth
                    cy = centerY / imageWidth
                    # width | height
                    width  = boxWidth / imageWidth
                    height = boxHeight / imageWidth
                    defaultBbox.append([cx, cy, width, height])

                    if maxSizes.__len__() > 0:
                        # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        boxWidth = boxHeight = math.sqrt(minSize * maxSize);
                        # cx | cy
                        cx = centerX / imageWidth
                        cy = centerY / imageWidth
                        # width | height
                        width = boxWidth / imageWidth
                        height = boxHeight / imageWidth
                        defaultBbox.append([cx, cy, width, height])

                    for ar in aspectRatio:
                        if ar == 1:
                            continue

                        boxWidth  = minSize * math.sqrt(ar)
                        boxHeight = minSize / math.sqrt(ar)

                        # cx | cy
                        cx = centerX / imageWidth
                        cy = centerY / imageWidth
                        # width | height
                        width = boxWidth / imageWidth
                        height = boxHeight / imageWidth
                        defaultBbox.append([cx, cy, width, height])

                    defaultBboxs.append(defaultBbox)

        # Convert default bboxs to numpy array
        defaultBboxs = numpy.asarray(defaultBboxs, dtype = 'float32')
        return defaultBboxs

    def ExtractFeature(self,
                       imsPath,
                       batchSize):
        ims = []
        numHasData = 0
        for imPath in imsPath:
            if imPath != '':
                extension = imPath.split('.')[-1]
                im = plt.imread(imPath, extension)
                im = skimage.transform.resize(im, (512, 512), preserve_range=True)
                im = im[:, :, [2, 1, 0]]
                im = numpy.transpose(im, (2, 0, 1))
                ims.append(im)
                numHasData += 1
        ims = numpy.asarray(ims, dtype = 'float32')

        VGG_MEAN = numpy.asarray([103.939, 116.779, 123.68], dtype = 'float32')
        VGG_MEAN = numpy.reshape(VGG_MEAN, (1, 3, 1, 1))

        ims = ims - VGG_MEAN

        if numHasData != 0:
            SSDfeatures = self.FeatureFunc(ims)
            feature     = SSDfeatures[0]
        else:
            numHasData = batchSize
            feature = numpy.zeros((batchSize, 5461, 256), dtype = 'float32')
        feature     = numpy.pad(feature, ((0, batchSize - numHasData), (0, 0), (0, 0)), mode = 'constant', constant_values = 0)
        return feature

    def LoadCaffeModel(self,
                       caffePrototxtPath,
                       caffeModelPath):
        self.Net.LoadCaffeModel(caffePrototxtPath, caffeModelPath)

    def LoadEncodeLayers(self,
                         conv4_3Path,
                         fc7Path,
                         conv6_2Path):
        file = open(conv4_3Path)
        self.Net.Layer['conv4_3_norm_encode'].LoadModel(file)
        file.close()

        file = open(fc7Path)
        self.Net.Layer['fc7_encode'].LoadModel(file)
        file.close()

        file = open(conv6_2Path)
        self.Net.Layer['conv6_2_encode'].LoadModel(file)
        file.close()



