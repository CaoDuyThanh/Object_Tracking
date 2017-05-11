import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class LSTMTrackingModel():
    def __init__(self,
                 numTruncate = 20,
                 numHidden   = 500,
                 inputsSize  = [576],
                 outputsSize = [1, 4]):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        FeaturesXGt   = T.matrix('FeaturesXGt', dtype = 'float32')
        FeaturesX     = T.tensor3('FeaturesX')
        TargetY       = T.tensor3('PredY')
        BboxY         = T.tensor3('BboxY')
        C             = T.vector('C', dtype = 'float32')
        S             = T.vector('S', dtype = 'float32')
        BoxsVariances = T.matrix('BoxsVariances')
        RatioPosNeg   = T.scalar('RatioPosNeg')

        # Create shared variable for input
        net = LSTMNet()
        net.NetName = 'LSTMTrackingNet'

        # Input
        # net.Layer['input']                 = InputLayer(net, X)
        net.LayerOpts['lstm_num_truncate'] = numTruncate
        # net.LayerOpts['reshape_new_shape'] = (net.LayerOpts['lstm_num_truncate'], 576)          # TODO: Need to set this size later
        # net.Layer['input_2d']              = ReshapeLayer(net, net.Layer['input'].Output)

        # Setting LSTM architecture
        net.LayerOpts['lstm_num_hidden']   = numHidden
        net.LayerOpts['lstm_inputs_size']  = inputsSize
        net.LayerOpts['lstm_outputs_size'] = outputsSize

        # Truncate lstm model
        currentC     = C
        currentS     = S
        preds         = []
        bboxs         = []
        predictLayers = []
        for truncId in range(net.LayerOpts['lstm_num_truncate']):
            # Create LSTM layer
            currentInput = FeaturesXGt[truncId]
            net.Layer['lstm_truncid_%d' % (truncId)] = LSTMLayer(net, currentInput, currentC, currentS)
            net.LayerOpts['lstm_params'] = net.Layer['lstm_truncid_%d' % (truncId)].Params

            # Predict next position based on current state
            currentInput = FeaturesX[truncId]
            tempLayer = LSTMLayer(net, currentInput, currentC, currentS)
            predictLayers.append(tempLayer)
            pred = SigmoidLayer(tempLayer.Output[0]).Output
            bbox = tempLayer.Output[1]
            preds.append(pred)
            bboxs.append(bbox)

            # Update stateS and stateC
            currentC = net.Layer['lstm_truncid_%d' % (truncId)].C
            currentS = net.Layer['lstm_truncid_%d' % (truncId)].S
        lastS = currentS
        lastC = currentC
        self.Net = net

        # Calculate cost function
        # Confidence loss
        cost = 0
        costPos = 0
        costLoc = 0
        costNeg = 0

        k0 = None
        k1 = None
        k2 = None
        k3 = None
        k4 = None
        for truncId in range(net.LayerOpts['lstm_num_truncate']):
            pred   = preds[truncId]
            bbox   = bboxs[truncId]
            target = TargetY[truncId]
            bboxgt = BboxY[truncId]

            numFeaturesPerIm   = pred.shape[0]
            numAnchorBoxPerLoc = pred.shape[1]

            pred   = pred.reshape((numFeaturesPerIm * numAnchorBoxPerLoc, 1))
            target = target.reshape((numFeaturesPerIm * numAnchorBoxPerLoc, 1))
            bbox   = bbox.reshape((numFeaturesPerIm * numAnchorBoxPerLoc, 4))
            bbox   = bbox / BoxsVariances
            bboxgt = bboxgt.reshape((numFeaturesPerIm * numAnchorBoxPerLoc, 4))

            allLocCost = T.sum(T.abs_(bbox - bboxgt), axis = 1, keepdims = True) * target

            allConfPosCost = - target * T.log(pred)
            allConfNegCost = - (1 - target) * T.log(1 - pred)

            allPosCost = allConfPosCost + allLocCost * 0
            allNegCost = allConfNegCost

            allPosCostSum = T.sum(allPosCost, axis = 1)
            allNegCostSum = T.sum(allNegCost, axis = 1)

            sortedPosCostIdx = T.argsort(allPosCostSum, axis = 0)
            sortedNegCostIdx = T.argsort(allNegCostSum, axis = 0)

            sortedPosCost = allPosCostSum[sortedPosCostIdx]
            sortedNegCost = allNegCostSum[sortedNegCostIdx]

            if k0 == None:
                k0 = target
            if k1 == None:
                k1 = allLocCost
            if k2 == None:
                k2 = pred
            if k3 == None:
                k3 = sortedPosCostIdx
            if k4 == None:
                k4 = sortedNegCostIdx

            numMax    = T.sum(T.neq(sortedPosCost, 0))
            # numNegMax = T.cast(T.floor(T.minimum(T.maximum(numMax * RatioPosNeg, 2), 300)), dtype = 'int32')
            numNegMax = T.cast(T.floor(numMax * RatioPosNeg), dtype = 'int32')

            top2PosCost = sortedPosCost[-numMax    : ]
            top6NegCost = sortedNegCost[-numNegMax : ]

            layerCost = (T.sum(top2PosCost) + T.sum(top6NegCost)) / numMax
            cost = cost + layerCost

            costPos = costPos + pred[sortedPosCostIdx[- numMax    : ]].mean()
            costLoc = costLoc + allLocCost.sum() / numMax
            costNeg = costNeg + pred[sortedNegCostIdx[- numNegMax : ]].mean()

        cost    = cost / net.LayerOpts['lstm_num_truncate']
        costPos = costPos / net.LayerOpts['lstm_num_truncate']
        costLoc = costLoc / net.LayerOpts['lstm_num_truncate']
        costNeg = costNeg / net.LayerOpts['lstm_num_truncate']

        # Create update function
        params = self.Net.Layer['lstm_truncid_0'].Params
        grads = T.grad(cost, params)
        updates = AdamGDUpdate(net, params=params, grads=grads).Updates

        # Train function
        self.TrainFunc = theano.function(inputs  = [FeaturesXGt, FeaturesX, TargetY, BboxY, S, C, BoxsVariances, RatioPosNeg],
                                         updates = updates,
                                         outputs = [cost, lastS, lastC, costPos, costLoc, costNeg, k0, k1, k2, k3, k4])

        self.PredFunc  = theano.function(inputs  = [FeaturesX, S, C],
                                         outputs = [preds[0], bboxs[0]])

        nextS = self.Net.Layer['lstm_truncid_0'].S
        nextC = self.Net.Layer['lstm_truncid_0'].C
        self.NextState = theano.function(inputs  = [FeaturesXGt, S, C],
                                         outputs = [nextS, nextC])

        #
        # self.TrainFunc1 = theano.function(inputs  = [FeaturesXGt, FeaturesX, TargetY, BboxY, S, C],
        #                                   outputs = temp1 + temp2 + temp3)

    def SaveModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].SaveModel(file)

    def LoadModel(self, file):
        # Save first layer
        self.Net.Layer['lstm_truncid_0'].LoadModel(file)


