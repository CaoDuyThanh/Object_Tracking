import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class LSTMTrackingModel():
    def __init__(self,
                 batchSize      = 1,
                 numTruncate    = (1, 1),
                 numHidden      = 512,
                 inputsSize     = [576],
                 outputsSize    = [1, 4],
                 featureFactory = None,
                 featureXBatch  = None,
                 YsBatch        = None,
                 BboxYsBatch    = None,
                 encodeCState   = None,
                 encodeHState   = None,
                 decodeCState   = None):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        self.FeatureXBatch = T.tensor4('FeatureXBatch') if featureXBatch is None else featureXBatch
        self.YsBatch       = T.tensor4('YsBatch') if YsBatch is None else YsBatch
        self.BboxYsBatch   = T.tensor4('BboxYsBatch') if BboxYsBatch is None else BboxYsBatch
        self.EncodeCState        = T.matrix('encodeCState', dtype='float32') if encodeCState is None else encodeCState
        self.EncodeHState        = T.matrix('encodeHState', dtype='float32') if encodeHState is None else encodeHState
        self.DecodeCState        = T.matrix('decodeCState', dtype='float32') if decodeCState is None else decodeCState
        self.FeatureFactory      = featureFactory

        self.FeatureEncodeXBatch = self.FeatureXBatch[:,  :numTruncate[0], :, :]
        self.FeatureDecodeXBatch = self.FeatureXBatch[:, 1:, :, :]
        self.EncodeYsBatch       = self.YsBatch[:,   :numTruncate[0], :, :]
        self.DecodeYsBatch       = self.YsBatch[:,  1:, :, :]
        self.BboxEncodeYsBatch   = self.BboxYsBatch[:,  :numTruncate[0], :, :]
        self.BboxDecodeYsBatch   = self.BboxYsBatch[:, 1:, :, :]

        # Create encoder LSTM layer
        self.EncodeNet         = LSTMNet()

        # Setting encoder architecture
        self.EncodeNet.NetName = 'LSTM_Encoder'
        self.EncodeNet.LayerOpts['lstm_num_truncate'] = numTruncate[0]
        self.EncodeNet.LayerOpts['lstm_num_hidden']   = numHidden
        self.EncodeNet.LayerOpts['lstm_inputs_size']  = [inputsSize[0]]
        self.EncodeNet.LayerOpts['lstm_outputs_size'] = []

        CStateEncodeBatch = []
        HStateEncodeBatch = []
        randomFactory     = T.shared_randomstreams.RandomStreams()
        for batchId in range(batchSize):
            # Truncate lstm model
            FeaturesX  = self.FeatureEncodeXBatch[batchId]
            Ys         = self.EncodeYsBatch[batchId]
            bboxYs     = self.BboxEncodeYsBatch[batchId]
            currentC   = self.EncodeCState[batchId]
            currentH   = self.EncodeHState[batchId]

            CStateEncodeSequence = []
            HStateEncodeSequence = []
            for truncId in range(self.EncodeNet.LayerOpts['lstm_num_truncate']):
                # Create LSTM cell
                featureX = FeaturesX[truncId]
                Y        = Ys[truncId]
                bboxY    = bboxYs[truncId]

                # sum6     = Y.sum()
                # idx      = (Y > 0).nonzero()[0]
                # featureX = theano.ifelse.ifelse(sum6 > 1., featureX[idx], featureX[:6] * 0)
                # bboxY    = theano.ifelse.ifelse(sum6 > 1., bboxY[idx], bboxY[:6] * 0)

                sum6 = Y.sum()
                idx1 = (Y[0 : 1444] > 0).nonzero()[0]
                rand = randomFactory.random_integers(size = (1,), low = 0, high = idx1.shape[0] - 1)
                idx1 = idx1[rand]

                idx2 = (Y[1444 : 1805] > 0).nonzero()[0] + 1444
                rand = randomFactory.random_integers(size=(1,), low=0, high=idx2.shape[0] - 1)
                idx2 = idx2[rand]

                idx3 = (Y[1805 : 1905] > 0).nonzero()[0] + 1805
                rand = randomFactory.random_integers(size=(1,), low=0, high=idx3.shape[0] - 1)
                idx3 = idx3[rand]

                idx4 = (Y[1905 : 1930] > 0).nonzero()[0] + 1905
                rand = randomFactory.random_integers(size=(1,), low=0, high=idx4.shape[0] - 1)
                idx4 = idx4[rand]

                idx5 = (Y[1930 : 1939] > 0).nonzero()[0] + 1930
                rand = randomFactory.random_integers(size=(1,), low=0, high=idx5.shape[0] - 1)
                idx5 = idx5[rand]

                idx6 = (Y[1939: 1940] > 0).nonzero()[0] + 1939
                rand = randomFactory.random_integers(size=(1,), low=0, high=idx6.shape[0] - 1)
                idx6 = idx6[rand]

                featureX = theano.ifelse.ifelse(sum6 > 1., featureX[[idx1, idx2, idx3, idx4, idx5, idx6]], featureX[[idx1, idx2, idx3, idx4, idx5, idx6]] * 0)
                bboxY    = theano.ifelse.ifelse(sum6 > 1., bboxY[[idx1, idx2, idx3, idx4, idx5, idx6]], bboxY[[idx1, idx2, idx3, idx4, idx5, idx6]] * 0)

                featureX  = featureX.reshape((T.prod(featureX.shape),))
                bboxY     = bboxY.reshape((T.prod(bboxY.shape),))

                featureXF = T.concatenate((featureX, bboxY), axis = 0)

                self.EncodeNet.Layer['lstm_truncid_%d' % (truncId)] = LSTMLayer(self.EncodeNet, featureXF, currentC, currentH)
                self.EncodeNet.LayerOpts['lstm_params'] = self.EncodeNet.Layer['lstm_truncid_%d' % (truncId)].Params

                # Update stateS and stateC
                currentC = self.EncodeNet.Layer['lstm_truncid_%d' % (truncId)].C
                currentH = self.EncodeNet.Layer['lstm_truncid_%d' % (truncId)].H

                CStateEncodeSequence.append(currentC)
                HStateEncodeSequence.append(currentH)
            CStateEncodeBatch.append(CStateEncodeSequence)
            HStateEncodeBatch.append(HStateEncodeSequence)
        lastCEncodeBatch = [cStateEncodeS[-1] for cStateEncodeS in CStateEncodeBatch]
        lastHEncodeBatch = [hStateEncodeS[-1] for hStateEncodeS in HStateEncodeBatch]


        # Create decoder
        self.DecodeNet = LSTMNet()

        # Setting encoder architecture
        self.DecodeNet.NetName = 'LSTM_Decoder'
        self.DecodeNet.LayerOpts['lstm_num_truncate'] = numTruncate[1]
        self.DecodeNet.LayerOpts['lstm_num_hidden']   = numHidden
        self.DecodeNet.LayerOpts['lstm_inputs_size']  = [256]
        self.DecodeNet.LayerOpts['lstm_outputs_size'] = outputsSize

        # CStateDecodeBatch = []
        # HStateDecodeBatch = []
        predBboxYsBatch  = []
        predYsBatch      = []
        for batchId in range(batchSize):
            # Truncate lstm model
            FeaturesX  = self.FeatureDecodeXBatch[batchId]
            currentC   = self.DecodeCState[batchId]
            currentHS  = HStateEncodeBatch[batchId]
            predBboxYs = []
            predYs     = []

            # CStateDecodeSequence = []
            # HStateDecodeSequence = []
            for truncId in range(self.EncodeNet.LayerOpts['lstm_num_truncate']):
                currentH = currentHS[truncId]

                # Create LSTM cell
                featureX = FeaturesX[truncId]
                self.DecodeNet.Layer['lstm_truncid_%d' % (truncId)] = LSTMLayer(self.DecodeNet, featureX, currentC, currentH)
                self.DecodeNet.LayerOpts['lstm_params'] = self.DecodeNet.Layer['lstm_truncid_%d' % (truncId)].Params

                # Predict next position based on current state
                predY     = SigmoidLayer(self.DecodeNet.Layer['lstm_truncid_%d' % (truncId)].Output[0]).Output
                predYs.append(predY)
                predBboxY = self.DecodeNet.Layer['lstm_truncid_%d' % (truncId)].Output[1]
                predBboxYs.append(predBboxY)

                # # Update stateS and stateC
                # currentC = self.DecodeNet.Layer['lstm_truncid_%d' % (truncId)].C
                # currentH = self.DecodeNet.Layer['lstm_truncid_%d' % (truncId)].H
                #
                # CStateDecodeSequence.append(currentC)
                # HStateDecodeSequence.append(currentH)

            # CStateDecodeBatch.append(CStateDecodeSequence)
            # HStateDecodeBatch.append(HStateDecodeSequence)
            predBboxYsBatch.append(predBboxYs)
            predYsBatch.append(predYs)

        # Calculate cost function
        # Confidence loss
        costBatch = 0
        posBatch  = 0
        negBatch  = 0
        sumTemp   = 0
        sumTemp1  = 0
        for batchId in range(batchSize):
            predYs       = predYsBatch[batchId]
            predBboxYs   = predBboxYsBatch[batchId]
            Ys           = self.DecodeYsBatch[batchId]
            bboxYs       = self.BboxDecodeYsBatch[batchId]
            costSequence = 0
            posSequence  = 0
            negSequence  = 0
            for truncId in range(self.EncodeNet.LayerOpts['lstm_num_truncate']):
                predY     = predYs[truncId]
                predBboxY = predBboxYs[truncId]
                Y         = Ys[truncId]
                bboxY     = bboxYs[truncId]

                allLocCost = T.sum(T.abs_(predBboxY - bboxY), axis = 1, keepdims = True) * Y

                allConfPosCost = - Y * T.log(predY)
                allConfNegCost = - (1 - Y) * T.log(1 - predY)

                allPosCost = allConfPosCost + allLocCost
                allNegCost = allConfNegCost

                allPosCostSum = T.sum(allPosCost, axis = 1)
                allNegCostSum = T.sum(allNegCost, axis = 1)

                sortedPosCostIdx = T.argsort(allPosCostSum, axis = 0)
                sortedNegCostIdx = T.argsort(allNegCostSum, axis = 0)

                sortedPosCost = allPosCostSum[sortedPosCostIdx]
                sortedNegCost = allNegCostSum[sortedNegCostIdx]

                numMax    = T.sum(T.neq(sortedPosCost, 0))
                numNegMax = T.cast(T.floor(numMax * 3), dtype = 'int32')

                top2PosCost = sortedPosCost[-numMax    : ]
                top6NegCost = sortedNegCost[-numNegMax : ]

                layerCost = T.where(numMax > 0, (T.sum(top2PosCost) + T.sum(top6NegCost)) / numMax, 0)
                sumTemp1 += T.where(numMax > 0, 1, 0)

                costSequence += layerCost

                sumTemp      += numMax
                posSequence  += T.where(numMax > 0, predY[sortedPosCostIdx[- numMax    : ]].sum(), 0)
                negSequence  += T.where(numMax > 0, predY[sortedNegCostIdx[- numNegMax : ]].sum(), 0)

            costBatch += costSequence
            posBatch  += posSequence
            negBatch  += negSequence
        costBatch /= sumTemp1
        posBatch  /= sumTemp
        negBatch  /= (sumTemp * 3)

        # Create update function
        params = self.EncodeNet.Layer['lstm_truncid_0'].Params + \
                 self.DecodeNet.Layer['lstm_truncid_0'].Params + \
                 featureFactory.Net.Layer['conv4_3_norm_encode'].Params + \
                 featureFactory.Net.Layer['fc7_encode'].Params + \
                 featureFactory.Net.Layer['conv6_2_encode'].Params
        grads = T.grad(costBatch, params)
        self.Optimizer = AdamGDUpdate(self.DecodeNet, params = params, grads = grads)
        updates = self.Optimizer.Updates

        # # Test function
        # self.TestFunc = theano.function(inputs  = [featureFactory.X,
        #                                             self.YsBatch,
        #                                            self.BboxYsBatch,],
        #                                  outputs = k)


        # Train function
        self.TrainFunc = theano.function(inputs  = [self.DecodeNet.NetOpts['learning_rate'],
                                                    featureFactory.X,
                                                    self.YsBatch,
                                                    self.BboxYsBatch,
                                                    self.EncodeCState,
                                                    self.EncodeHState,
                                                    self.DecodeCState],
                                         updates = updates,
                                         outputs = [costBatch] + lastCEncodeBatch + lastHEncodeBatch + \
                                                                 [posBatch, negBatch])

        # Valid function
        self.ValidFunc = theano.function(inputs=[featureFactory.X,
                                                 self.YsBatch,
                                                 self.BboxYsBatch,
                                                 self.EncodeCState,
                                                 self.EncodeHState,
                                                 self.DecodeCState],
                                         outputs=[costBatch] + lastCEncodeBatch + lastHEncodeBatch + \
                                                 [posBatch, negBatch])

        # Pred function
        self.PredFunc = theano.function(inputs = [featureFactory.X,
                                                  self.YsBatch,
                                                  self.BboxYsBatch,
                                                  self.EncodeCState,
                                                  self.EncodeHState,
                                                  self.DecodeCState],
                                        outputs=[predYsBatch[0][0], predBboxYsBatch[0][0]] + lastCEncodeBatch + lastHEncodeBatch)


        # self.PredFunc  = theano.function(inputs  = [FeaturesX, S, C],
        #                                  outputs = [preds[0], bboxs[0]])
        #
        # nextS = self.Net.Layer['lstm_truncid_0'].S
        # nextC = self.Net.Layer['lstm_truncid_0'].C
        # self.NextState = theano.function(inputs  = [FeaturesXGt, S, C],
        #                                  outputs = [nextS, nextC])

        #
        # self.TrainFunc1 = theano.function(inputs  = [FeaturesXGt, FeaturesX, TargetY, BboxY, S, C],
        #                                   outputs = temp1 + temp2 + temp3)

    def SaveModel(self, file):
        # Save first layer
        self.EncodeNet.Layer['lstm_truncid_0'].SaveModel(file)
        self.DecodeNet.Layer['lstm_truncid_0'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv4_3_norm_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['fc7_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv6_2_encode'].SaveModel(file)

    def SaveState(self, file):
        self.EncodeNet.Layer['lstm_truncid_0'].SaveModel(file)
        self.DecodeNet.Layer['lstm_truncid_0'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv4_3_norm_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['fc7_encode'].SaveModel(file)
        self.FeatureFactory.Net.Layer['conv6_2_encode'].SaveModel(file)
        self.Optimizer.SaveModel(file)

    def LoadModel(self, file):
        # Save first layer
        self.EncodeNet.Layer['lstm_truncid_0'].LoadModel(file)
        self.DecodeNet.Layer['lstm_truncid_0'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv4_3_norm_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['fc7_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv6_2_encode'].LoadModel(file)

    def LoadState(self, file):
        # Save first layer
        self.EncodeNet.Layer['lstm_truncid_0'].LoadModel(file)
        self.DecodeNet.Layer['lstm_truncid_0'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv4_3_norm_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['fc7_encode'].LoadModel(file)
        self.FeatureFactory.Net.Layer['conv6_2_encode'].LoadModel(file)
        self.Optimizer.LoadModel(file)
