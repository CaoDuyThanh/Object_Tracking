import theano
import theano.tensor as T
import numpy
import cv2
from Layers.LayerHelper import *
from Layers.Net import *

BATCH_SIZE = 1

class SSDModel():
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
        net.LayerOpts['reshape_new_shape'] = (net.NetOpts['batch_size'], 3, 300, 300)
        net.Layer['input_4d'] = ReshapeLayer(net, net.Layer['input'].Output)

        net.LayerOpts['pool_boder_mode'] = 1
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

        # conv4_3_norm_mbox_conf
        net.LayerOpts['conv2D_filter_shape'] = (84, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv4_3_norm_mbox_conf_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_norm_mbox_conf_b'
        net.Layer['conv4_3_norm_mbox_conf'] = ConvLayer(net, net.Layer['conv4_3_norm'].Output)

        net.LayerOpts['permute_dimension']       = (0, 2, 3, 1)
        net.Layer['conv4_3_norm_mbox_conf_perm'] = PermuteLayer(net, net.Layer['conv4_3_norm_mbox_conf'].Output)
        net.LayerOpts['flatten_ndim']            = 2
        net.Layer['conv4_3_norm_mbox_conf_flat'] = FlattenLayer(net, net.Layer['conv4_3_norm_mbox_conf_perm'].Output)

        # conv4_3_norm_mbox_loc
        net.LayerOpts['conv2D_filter_shape'] = (16, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv4_3_norm_mbox_loc_W'
        net.LayerOpts['conv2D_bName'] = 'conv4_3_norm_mbox_loc_b'
        net.Layer['conv4_3_norm_mbox_loc'] = ConvLayer(net, net.Layer['conv4_3_norm'].Output)

        net.LayerOpts['permute_dimension']      = (0, 2, 3, 1)
        net.Layer['conv4_3_norm_mbox_loc_perm'] = PermuteLayer(net, net.Layer['conv4_3_norm_mbox_loc'].Output)
        net.LayerOpts['flatten_ndim']           = 2
        net.Layer['conv4_3_norm_mbox_loc_flat'] = FlattenLayer(net, net.Layer['conv4_3_norm_mbox_loc_perm'].Output)

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
        # fc7_mbox_conf
        net.LayerOpts['conv2D_filter_shape'] = (126, 1024, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'fc7_mbox_conf_W'
        net.LayerOpts['conv2D_bName']        = 'fc7_mbox_conf_b'
        net.Layer['fc7_mbox_conf']  = ConvLayer(net, net.Layer['relu7'].Output)

        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['fc7_mbox_conf_perm']    = PermuteLayer(net, net.Layer['fc7_mbox_conf'].Output)
        net.LayerOpts['flatten_ndim']      = 2
        net.Layer['fc7_mbox_conf_flat']    = FlattenLayer(net, net.Layer['fc7_mbox_conf_perm'].Output)

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

        # fc7_mbox_loc
        net.LayerOpts['conv2D_filter_shape'] = (24, 1024, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'fc7_mbox_loc_W'
        net.LayerOpts['conv2D_bName']        = 'fc7_mbox_loc_b'
        net.Layer['fc7_mbox_loc'] = ConvLayer(net, net.Layer['relu7'].Output)

        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['fc7_mbox_loc_perm']     = PermuteLayer(net, net.Layer['fc7_mbox_loc'].Output)
        net.LayerOpts['flatten_ndim']      = 2
        net.Layer['fc7_mbox_loc_flat']     = FlattenLayer(net, net.Layer['fc7_mbox_loc_perm'].Output)

        # fc7_mbox_priorbox
        # net.Layer['fc7_mbox_priorbox']


        # Second sub convolution to get predicted box
        # conv6_2_mbox_conf
        net.LayerOpts['conv2D_filter_shape'] = (126, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'conv6_2_mbox_conf_W'
        net.LayerOpts['conv2D_bName']        = 'conv6_2_mbox_conf_b'
        net.Layer['conv6_2_mbox_conf'] = ConvLayer(net, net.Layer['conv6_2_relu'].Output)

        net.LayerOpts['permute_dimension']  = (0, 2, 3, 1)
        net.Layer['conv6_2_mbox_conf_perm'] = PermuteLayer(net, net.Layer['conv6_2_mbox_conf'].Output)
        net.LayerOpts['flatten_ndim']       = 2
        net.Layer['conv6_2_mbox_conf_flat'] = FlattenLayer(net, net.Layer['conv6_2_mbox_conf_perm'].Output)

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

        # conv6_2_mbox_loc
        net.LayerOpts['conv2D_filter_shape'] = (24, 512, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'conv6_2_mbox_loc_W'
        net.LayerOpts['conv2D_bName']        = 'conv6_2_mbox_loc_b'
        net.Layer['conv6_2_mbox_loc'] = ConvLayer(net, net.Layer['conv6_2_relu'].Output)

        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv6_2_mbox_loc_perm'] = PermuteLayer(net, net.Layer['conv6_2_mbox_loc'].Output)
        net.LayerOpts['flatten_ndim']      = 2
        net.Layer['conv6_2_mbox_loc_flat'] = FlattenLayer(net, net.Layer['conv6_2_mbox_loc_perm'].Output)

        # fc7_mbox_priorbox
        # net.Layer['fc7_mbox_priorbox']



        # Third sub convolution to get predicted box
        # conv7_2_mbox_conf
        net.LayerOpts['conv2D_filter_shape'] = (126, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'conv7_2_mbox_conf_W'
        net.LayerOpts['conv2D_bName']        = 'conv7_2_mbox_conf_b'
        net.Layer['conv7_2_mbox_conf'] = ConvLayer(net, net.Layer['conv7_2_relu'].Output)

        net.LayerOpts['permute_dimension']  = (0, 2, 3, 1)
        net.Layer['conv7_2_mbox_conf_perm'] = PermuteLayer(net, net.Layer['conv7_2_mbox_conf'].Output)
        net.LayerOpts['flatten_ndim']       = 2
        net.Layer['conv7_2_mbox_conf_flat'] = FlattenLayer(net, net.Layer['conv7_2_mbox_conf_perm'].Output)

        # conv8_1 and conv8_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'conv8_1_W'
        net.LayerOpts['conv2D_bName']        = 'conv8_1_b'
        net.Layer['conv8_1']      = ConvLayer(net, net.Layer['conv7_2_relu'].Output)
        net.Layer['conv8_1_relu'] = ReLULayer(net.Layer['conv8_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'conv8_2_W'
        net.LayerOpts['conv2D_bName']        = 'conv8_2_b'
        net.Layer['conv8_2'] = ConvLayer(net, net.Layer['conv8_1_relu'].Output)
        net.Layer['conv8_2_relu'] = ReLULayer(net.Layer['conv8_2'].Output)

        # conv7_2_mbox_loc
        net.LayerOpts['conv2D_filter_shape'] = (24, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv7_2_mbox_loc_W'
        net.LayerOpts['conv2D_bName'] = 'conv7_2_mbox_loc_b'
        net.Layer['conv7_2_mbox_loc'] = ConvLayer(net, net.Layer['conv7_2_relu'].Output)

        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv7_2_mbox_loc_perm'] = PermuteLayer(net, net.Layer['conv7_2_mbox_loc'].Output)
        net.LayerOpts['flatten_ndim']      = 2
        net.Layer['conv7_2_mbox_loc_flat'] = FlattenLayer(net, net.Layer['conv7_2_mbox_loc_perm'].Output)

        # fc7_mbox_priorbox
        # net.Layer['fc7_mbox_priorbox']


        # Fourth sub convolution to get predicted box
        # conv8_2_mbox_conf
        net.LayerOpts['conv2D_filter_shape'] = (84, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName']        = 'conv8_2_mbox_conf_W'
        net.LayerOpts['conv2D_bName']        = 'conv8_2_mbox_conf_b'
        net.Layer['conv8_2_mbox_conf'] = ConvLayer(net, net.Layer['conv8_2_relu'].Output)

        net.LayerOpts['permute_dimension']  = (0, 2, 3, 1)
        net.Layer['conv8_2_mbox_conf_perm'] = PermuteLayer(net, net.Layer['conv8_2_mbox_conf'].Output)
        net.LayerOpts['flatten_ndim']       = 2
        net.Layer['conv8_2_mbox_conf_flat'] = FlattenLayer(net, net.Layer['conv8_2_mbox_conf_perm'].Output)

        # conv9_1 and conv9_2
        net.LayerOpts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName']        = 'conv9_1_W'
        net.LayerOpts['conv2D_bName']        = 'conv9_1_b'
        net.Layer['conv9_1']      = ConvLayer(net, net.Layer['conv8_2_relu'].Output)
        net.Layer['conv9_1_relu'] = ReLULayer(net.Layer['conv9_1'].Output)

        net.LayerOpts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = 0
        net.LayerOpts['conv2D_WName'] = 'conv9_2_W'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_b'
        net.Layer['conv9_2']      = ConvLayer(net, net.Layer['conv9_1_relu'].Output)
        net.Layer['conv9_2_relu'] = ReLULayer(net.Layer['conv9_2'].Output)

        # conv8_2_mbox_loc
        net.LayerOpts['conv2D_filter_shape'] = (16, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv8_2_mbox_loc_W'
        net.LayerOpts['conv2D_bName'] = 'conv8_2_mbox_loc_b'
        net.Layer['conv8_2_mbox_loc'] = ConvLayer(net, net.Layer['conv8_2_relu'].Output)

        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv8_2_mbox_loc_perm'] = PermuteLayer(net, net.Layer['conv8_2_mbox_loc'].Output)
        net.LayerOpts['flatten_ndim']      = 2
        net.Layer['conv8_2_mbox_loc_flat'] = FlattenLayer(net, net.Layer['conv8_2_mbox_loc_perm'].Output)

        # fc7_mbox_priorbox
        # net.Layer['fc7_mbox_priorbox']


        # Fifth sub convolution to get predicted box
        # conv9_2_mbox_conf
        net.LayerOpts['conv2D_filter_shape'] = (84, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv9_2_mbox_conf_W'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_mbox_conf_b'
        net.Layer['conv9_2_mbox_conf'] = ConvLayer(net, net.Layer['conv9_2_relu'].Output)

        net.LayerOpts['permute_dimension']  = (0, 2, 3, 1)
        net.Layer['conv9_2_mbox_conf_perm'] = PermuteLayer(net, net.Layer['conv9_2_mbox_conf'].Output)
        net.LayerOpts['flatten_ndim']       = 2
        net.Layer['conv9_2_mbox_conf_flat'] = FlattenLayer(net, net.Layer['conv9_2_mbox_conf_perm'].Output)

        # conv9_2_mbox_loc
        net.LayerOpts['conv2D_filter_shape'] = (16, 256, 3, 3)
        net.LayerOpts['conv2D_stride']       = (1, 1)
        net.LayerOpts['conv2D_border_mode']  = (1, 1)
        net.LayerOpts['conv2D_WName'] = 'conv9_2_mbox_loc_W'
        net.LayerOpts['conv2D_bName'] = 'conv9_2_mbox_loc_b'
        net.Layer['conv9_2_mbox_loc'] = ConvLayer(net, net.Layer['conv9_2_relu'].Output)

        net.LayerOpts['permute_dimension'] = (0, 2, 3, 1)
        net.Layer['conv9_2_mbox_loc_perm'] = PermuteLayer(net, net.Layer['conv9_2_mbox_loc'].Output)
        net.LayerOpts['flatten_ndim']      = 2
        net.Layer['conv9_2_mbox_loc_flat'] = FlattenLayer(net, net.Layer['conv9_2_mbox_loc_perm'].Output)

        # fc7_mbox_priorbox
        # net.Layer['fc7_mbox_priorbox']

        # Concat mbox_conf and mbox_loc
        net.Layer['mbox_conf'] = ConcatLayer(net, [net.Layer['conv4_3_norm_mbox_conf_flat'].Output,
                                                   net.Layer['fc7_mbox_conf_flat'].Output,
                                                   net.Layer['conv6_2_mbox_conf_flat'].Output,
                                                   net.Layer['conv7_2_mbox_conf_flat'].Output,
                                                   net.Layer['conv8_2_mbox_conf_flat'].Output,
                                                   net.Layer['conv9_2_mbox_conf_flat'].Output])
        net.Layer['mbox_loc']  = ConcatLayer(net, [net.Layer['conv4_3_norm_mbox_loc_flat'].Output,
                                                   net.Layer['fc7_mbox_loc_flat'].Output,
                                                   net.Layer['conv6_2_mbox_loc_flat'].Output,
                                                   net.Layer['conv7_2_mbox_loc_flat'].Output,
                                                   net.Layer['conv8_2_mbox_loc_flat'].Output,
                                                   net.Layer['conv9_2_mbox_loc_flat'].Output])

        net.LayerOpts['reshape_new_shape'] = (8732, 21)
        net.Layer['mbox_conf_reshape']     = ReshapeLayer(net, net.Layer['mbox_conf'].Output)

        net.Layer['mbox_conf_softmax'] = SoftmaxLayer(net.Layer['mbox_conf_reshape'].Output)

        net.LayerOpts['reshape_new_shape'] = (8732, 4)
        net.Layer['mbox_loc_flatten']      = ReshapeLayer(net, net.Layer['mbox_loc'].Output)

        self.Net = net

        # Predict function
        label = T.argmax(net.Layer['mbox_conf_softmax'].Output, axis = 1)
        self.PredFunc = theano.function(
                            inputs  = [X],
                            outputs = [label,
                                       net.Layer['mbox_loc_flatten'].Output])

        self.Layers     = []
        self.LayersName = []
        for name, layer in sorted(net.Layer.items()):
            self.LayersName.append(name)
            self.Layers.append(layer.Output)

        self.TestFunc = theano.function(
                            inputs = [X],
                            outputs = self.Layers
        )

        # self.ProbFunc = theano.function(
        #                     inputs  = [X],
        #                     outputs = net.Layer['prob'].Output)

    def LoadCaffeModel(self,
                  caffePrototxtPath,
                  caffeModelPath):
        self.Net.LoadCaffeModel(caffePrototxtPath, caffeModelPath)

    def TestNetwork(self,
                    im):
        return self.PredFunc(im)
