import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class LSTMTrackingModel():
    def __init__(self,
                 batchSize   = 5,
                 numTruncate = 20,
                 numHidden   = 500,
                 inputsSize  = [576],
                 outputsSize = [1]):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        FeaturesXBatch = T.tensor4('FeaturesXBatch', dtype = 'float32')
        BboxXsBatch    = T.tensor4('BboxXsBatch', dtype = 'float32')
        BboxYsBatch    = T.tensor4('BboxYsBatch', dtype = 'float32')
        CBatch         = T.matrix('CBatch', dtype = 'float32')
        SBatch         = T.matrix('SBatch', dtype = 'float32')

        # FeaturesX     = T.tensor3('FeaturesX', dtype = 'float32')
        # BboxXs        = T.tensor3('BboxXs', dtype = 'float32')
        # BboxYs        = T.tensor3('BboxYs', dtype = 'float32')
        # C             = T.vector('C', dtype = 'float32')
        # S             = T.vector('S', dtype = 'float32')

        # Create shared variable for input
        net = LSTMNet()
        net.NetName = 'LSTMTrackingNet'

        # Input
        net.LayerOpts['lstm_num_truncate'] = numTruncate

        # Setting LSTM architecture
        net.LayerOpts['lstm_num_hidden']   = numHidden
        net.LayerOpts['lstm_inputs_size']  = inputsSize
        net.LayerOpts['lstm_outputs_size'] = outputsSize


        predBboxYsBatch = []
        lastSBatch      = []
        lastCBatch      = []
        for batchId in range(batchSize):
            # Truncate lstm model
            FeaturesX    = FeaturesXBatch[batchId]
            BboxXs       = BboxXsBatch[batchId]
            currentC     = CBatch[batchId]
            currentS     = SBatch[batchId]
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

            predBboxYsBatch.append(predBboxYs)
            lastSBatch.append(lastS)
            lastCBatch.append(lastC)
        self.Net = net

        # Calculate cost function
        # Confidence loss
        costBatch = 0
        for batchId in range(batchSize):
            predBboxYs = predBboxYsBatch[batchId]
            BboxYs     = BboxYsBatch[batchId]
            costSequence = 0
            for truncId in range(net.LayerOpts['lstm_num_truncate']):
                predBboxY   = predBboxYs[truncId]
                bboxY       = BboxYs[truncId]
                bboxCost    = T.sum(- bboxY * T.log(predBboxY) - (1 - bboxY) * T.log(1- predBboxY))
                costSequence += bboxCost
            costSequence /= net.LayerOpts['lstm_num_truncate']
            costBatch += costSequence
        costBatch = costBatch / batchSize

        # Create update function
        params = self.Net.Layer['lstm_truncid_0'].Params
        grads = T.grad(costBatch, params)
        updates = AdamGDUpdate(net, params = params, grads = grads).Updates

        # Train function
        self.TrainFunc = theano.function(inputs  = [FeaturesXBatch, BboxXsBatch, BboxYsBatch, SBatch, CBatch],
                                         updates = updates,
                                         outputs = [costBatch] + lastSBatch + lastCBatch)

        # self.PredFunc  = theano.function(inputs  = [FeaturesX, S, C],
        #                                  outputs = [preds[0], bboxs[0]])

        nextS = self.Net.Layer['lstm_truncid_0'].SMean
        nextC = self.Net.Layer['lstm_truncid_0'].CMean
        self.NextState = theano.function(inputs  = [FeaturesXBatch, BboxXsBatch, SBatch, CBatch],
                                         outputs = [predBboxYsBatch[0][0], nextS, nextC])

    def SaveModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].SaveModel(file)

    def LoadModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].LoadModel(file)