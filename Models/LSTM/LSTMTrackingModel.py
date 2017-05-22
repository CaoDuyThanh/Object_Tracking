import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class LSTMTrackingModel():
    def __init__(self,
                 featureFactory = None,
                 batchSize      = 5,
                 numTruncate    = 20,
                 numHidden      = 500,
                 inputsSize     = [576],
                 outputsSize    = [1]):
        ####################################
        #       Create model               #
        ####################################

        self.FeatureFactory = featureFactory

        # Extract features using feature factory
        features = featureFactory.Net.Layer['features'].Output
        self.FeaturesXBatch = features.reshape((batchSize, numTruncate, 5461, 1))

        # Create tensor variables to store input / output data
        # self.FeaturesXBatch = T.tensor4('FeaturesXBatch', dtype = 'float32')
        self.BboxXsBatch    = T.tensor4('BboxXsBatch', dtype = 'float32')
        self.BboxYsBatch    = T.tensor4('BboxYsBatch', dtype = 'float32')
        self.CBatch         = T.matrix('CBatch', dtype = 'float32')
        self.SBatch         = T.matrix('SBatch', dtype = 'float32')

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
            FeaturesX    = self.FeaturesXBatch[batchId]
            BboxXs       = self.BboxXsBatch[batchId]
            currentC     = self.CBatch[batchId]
            currentS     = self.SBatch[batchId]
            predBboxYs   = []
            # predictLayers = []
            for truncId in range(net.LayerOpts['lstm_num_truncate']):
                # Create LSTM layer
                featureX     = FeaturesX[truncId]
                bboxX        = BboxXs[truncId]
                # Concat feature and bbox into one input feature
                net.LayerOpts['concatenate_axis'] = 0
                currentInput = ConcatLayer(net = net,
                                           inputs = [featureX, bboxX]).Output
                currentInput = currentInput.T
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
        posBatch  = 0
        negBatch  = 0
        for batchId in range(batchSize):
            predBboxYs = predBboxYsBatch[batchId]
            BboxYs     = self.BboxYsBatch[batchId]
            costSequence = 0
            posSequence  = 0
            negSequence  = 0
            for truncId in range(net.LayerOpts['lstm_num_truncate']):
                predBboxY   = predBboxYs[truncId]
                bboxY       = BboxYs[truncId].T
                numPos      = T.maximum(1, T.sum(T.neq(bboxY, 0)))
                numNeg      = T.maximum(1, T.sum(T.neq(bboxY, 1)))
                # bboxCost    = T.sum(- bboxY * T.log(predBboxY) - (1 - bboxY) * T.log(1- predBboxY))

                posSequence += T.sum(- bboxY * T.log(predBboxY)) / numPos
                negSequence += T.sum(- (1 - bboxY) * T.log(1 - predBboxY)) / numNeg
                costSequence += T.sum(- bboxY * T.log(predBboxY)) / numPos + \
                                T.sum(- (1 - bboxY) * T.log(1 - predBboxY)) / numNeg

            costSequence /= net.LayerOpts['lstm_num_truncate']
            posSequence /= net.LayerOpts['lstm_num_truncate']
            negSequence /= net.LayerOpts['lstm_num_truncate']
            costBatch += costSequence
            posBatch += posSequence
            negBatch += negSequence
        costBatch = costBatch / batchSize
        posBatch = posBatch / batchSize
        negBatch = negBatch / batchSize

        # Create update function
        featureParams =  featureFactory.Net.Layer['conv4_3_norm_encode'].Params +\
                         featureFactory.Net.Layer['fc7_encode'].Params +\
                         featureFactory.Net.Layer['conv6_2_encode'].Params +\
                         featureFactory.Net.Layer['conv7_2_encode'].Params +\
                         featureFactory.Net.Layer['conv8_2_encode'].Params +\
                         featureFactory.Net.Layer['conv9_2_encode'].Params +\
                         featureFactory.Net.Layer['conv10_2_encode'].Params
        params = self.Net.Layer['lstm_truncid_0'].Params + featureParams
        grads = T.grad(costBatch, params)
        updates = AdamGDUpdate(net, params = params, grads = grads).Updates

        # Train function
        self.TrainFunc = theano.function(inputs  = [featureFactory.X,
                                                    self.BboxXsBatch,
                                                    self.BboxYsBatch,
                                                    self.SBatch,
                                                    self.CBatch],
                                         updates = updates,
                                         outputs = [costBatch] + lastSBatch + lastCBatch + [posBatch, negBatch])

        # Valid function
        self.ValidFunc = theano.function(inputs=[featureFactory.X,
                                                 self.BboxXsBatch,
                                                 self.BboxYsBatch,
                                                 self.SBatch,
                                                 self.CBatch],
                                         outputs=[costBatch] + lastSBatch + lastCBatch + [posBatch, negBatch])

        nextS = self.Net.Layer['lstm_truncid_0'].SMean
        nextC = self.Net.Layer['lstm_truncid_0'].CMean
        self.NextState = theano.function(inputs  = [featureFactory.X,
                                                    self.BboxXsBatch,
                                                    self.SBatch,
                                                    self.CBatch],
                                         outputs = [predBboxYsBatch[0][0].T, nextS, nextC])

    def SaveModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv4_3_norm_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['fc7_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv6_2_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv7_2_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv8_2_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv9_2_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv10_2_encode'].SaveModel(file)

    def LoadModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv4_3_norm_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['fc7_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv6_2_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv7_2_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv8_2_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv9_2_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv10_2_encode'].LoadModel(file)
