import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class LSTMTrackingModel():
    def __init__(self):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        X = T.tensor4('X')
        Y = T.ivector('Y')
        C = T.fvector('C')
        S = T.fvector('S')

        # Create shared variable for input
        net = LSTMNet()
        net.NetName = 'LSTMTrackingNet'

        # Input
        net.Layer['input']                 = InputLayer(net, X)
        net.LayerOpts['lstm_num_truncate'] = 20
        net.LayerOpts['reshape_new_shape'] = (net.LayerOpts['lstm_num_truncate'], 300) # TODO: Need to set this size later
        net.Layer['input_2d']              = ReshapeLayer(net, net.Layer['input'].Output)

        # Truncate lstm model
        currentC     = C
        currentS     = S
        for truncId in range(net.LayerOpts['lstm_num_truncate']):
            currentInput = net.Layer['input_2d'].Output[truncId]
            net.Layer['lstm_truncid_%d' % (truncId)] = LSTMLayer(net, currentInput, currentC, currentS)
            currentC = net.Layer['lstm_truncid_%d' % (truncId)].C
            currentS = net.Layer['lstm_truncid_%d' % (truncId)].S

        self.Net = net

        # Predict function
        label = T.argmax(net.Layer['mbox_conf_softmax'].Output, axis=1)
        self.PredFunc = theano.function(
            inputs=[X],
            outputs=[label,
                     net.Layer['mbox_loc_flatten'].Output])

