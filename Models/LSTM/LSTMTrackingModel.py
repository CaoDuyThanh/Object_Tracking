import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class LSTMTrackingModel():
    def __init__(self,
                 numTruncate = 20,
                 numHidden   = 500,
                 inputsSize  = [576],
                 outputsSize = [1]):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        FeaturesX     = T.tensor3('FeaturesX', dtype = 'float32')
        BboxXs        = T.tensor3('BboxXs', dtype = 'float32')
        BboxYs        = T.tensor3('BboxYs', dtype = 'float32')
        C             = T.vector('C', dtype = 'float32')
        S             = T.vector('S', dtype = 'float32')

        # Create shared variable for input
        net = LSTMNet()
        net.NetName = 'LSTMTrackingNet'

        # Input
        net.LayerOpts['lstm_num_truncate'] = numTruncate

        # Setting LSTM architecture
        net.LayerOpts['lstm_num_hidden']   = numHidden
        net.LayerOpts['lstm_inputs_size']  = inputsSize
        net.LayerOpts['lstm_outputs_size'] = outputsSize

        # Truncate lstm model
        currentC     = C
        currentS     = S
        predBboxYs   = []
        # predictLayers = []
        for truncId in range(net.LayerOpts['lstm_num_truncate']):
            # Create LSTM layer
            featureX     = FeaturesX[truncId]
            bboxX        = BboxXs[truncId]
            # Concat feature and bbox into one input feature
            net.LayerOpts['concatenate_axis'] = 1
            currentInput = ConcatLayer(net = net,
                                       inputs = [featureX, bboxX]).Output
            net.Layer['lstm_truncid_%d' % (truncId)] = LSTMLayer(net, currentInput, currentC, currentS)
            net.LayerOpts['lstm_params']             = net.Layer['lstm_truncid_%d' % (truncId)].Params

            # Predict next position based on current state
            predBboxY = SigmoidLayer(net.Layer['lstm_truncid_%d' % (truncId)].Output[0]).Output
            predBboxYs.append(predBboxY)

            # Update stateS and stateC
            currentC = net.Layer['lstm_truncid_%d' % (truncId)].CMean
            currentS = net.Layer['lstm_truncid_%d' % (truncId)].SMean
        lastS = currentS
        lastC = currentC
        self.Net = net

        # Calculate cost function
        # Confidence loss
        cost = 0
        for truncId in range(net.LayerOpts['lstm_num_truncate']):
            predBboxY   = predBboxYs[truncId]
            bboxY       = BboxYs[truncId]
            bboxCost    = T.sum(T.sqr(predBboxY - bboxY))
            cost = cost + bboxCost
        cost    = cost / net.LayerOpts['lstm_num_truncate']

        # Create update function
        params = self.Net.Layer['lstm_truncid_0'].Params
        grads = T.grad(cost, params)
        updates = AdamGDUpdate(net, params = params, grads = grads).Updates

        # Train function
        self.TrainFunc = theano.function(inputs  = [FeaturesX, BboxXs, BboxYs, S, C],
                                         updates = updates,
                                         outputs = [cost, lastS, lastC])

        # self.PredFunc  = theano.function(inputs  = [FeaturesX, S, C],
        #                                  outputs = [preds[0], bboxs[0]])

        nextS = self.Net.Layer['lstm_truncid_0'].S
        nextC = self.Net.Layer['lstm_truncid_0'].C
        self.NextState = theano.function(inputs  = [FeaturesX, BboxXs, S, C],
                                         outputs = [nextS, nextC])

    def SaveModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].SaveModel(file)

    def LoadModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].LoadModel(file)