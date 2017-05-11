import theano.tensor as T
import pickle
import cPickle
from UtilLayer import *
from BasicLayer import *

class LSTMLayer(BasicLayer):
    def __init__(self,
                 net,
                 inputs,
                 ckm1,
                 skm1):
        # Save all information to its layer
        self.NumHidden   = net.LayerOpts['lstm_num_hidden']
        self.InputsSize  = net.LayerOpts['lstm_inputs_size']
        self.OutputsSize = net.LayerOpts['lstm_outputs_size']
        self.Params      = net.LayerOpts['lstm_params']

        if self.Params is None:
            # Parameters for list of input
            Uis = []; Ufs = []; Ucs = []; Uos = []            # Input Weight
            for idx, inputSize in enumerate(self.InputsSize):
                Ui = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 1, 'lstm_Ui_%d' % (idx))
                Uf = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 1, 'lstm_Uf_%d' % (idx))
                Uc = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 1, 'lstm_Uc_%d' % (idx))
                Uo = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 1, 'lstm_Uo_%d' % (idx))

                Uis.append(Ui)
                Ufs.append(Uf)
                Ucs.append(Uc)
                Uos.append(Uo)

            # Init Wi | Ui | bi
            Wi = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, 'lstm_Wi')
            bi = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, 'lstm_bi')

            # Init Wf | bf
            Wf = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, 'lstm_Wf')
            bf = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, 'lstm_bf')

            # Init Wc | bc
            Wc = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, 'lstm_Wc')
            bc = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, 'lstm_bc')

            # Init Wo | bo
            Wo = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, 'lstm_Wo')
            bo = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, 'lstm_bo')

            # Parameters for list of output
            Wys = [];  bys = [];   # Output Weight | Bias
            for idx, outputSize in enumerate(self.OutputsSize):
                Wy = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, outputSize), 1, 'lstm_Wy_%d' % (idx))
                by = CreateSharedParameter(net.NetOpts['rng'], (outputSize,), 0, 'lstm_by_%d' % (idx))

                Wys.append(Wy)
                bys.append(by)

            self.Params = Uis + Ufs + Ucs + Uos + \
                          Wys + bys + \
                          [Wi,  Wf,   Wc,   Wo] + \
                          [bi,  bf,   bc,   bo]

        # Get all weight from param
        numInputs  = self.InputsSize.__len__()
        numOutputs = self.OutputsSize.__len__()
        Uis = self.Params[0 : numInputs]
        Ufs = self.Params[numInputs * 1 : numInputs * 2]
        Ucs = self.Params[numInputs * 2 : numInputs * 3]
        Uos = self.Params[numInputs * 3 : numInputs * 4]
        Wys = self.Params[numInputs * 4                  : numInputs * 4 + numOutputs * 1]
        bys = self.Params[numInputs * 4 + numOutputs * 1 : numInputs * 4 + numOutputs * 2]
        Wi  = self.Params[numInputs * 4 + numOutputs * 2]
        Wf  = self.Params[numInputs * 4 + numOutputs * 2 + 1]
        Wc  = self.Params[numInputs * 4 + numOutputs * 2 + 2]
        Wo  = self.Params[numInputs * 4 + numOutputs * 2 + 3]
        bi  = self.Params[numInputs * 4 + numOutputs * 2 + 4]
        bf  = self.Params[numInputs * 4 + numOutputs * 2 + 5]
        bc  = self.Params[numInputs * 4 + numOutputs * 2 + 6]
        bo  = self.Params[numInputs * 4 + numOutputs * 2 + 7]

        # Slice inputs to smaller input
        # slicedInput = []; startId = 0
        # for inputSize in self.InputsSize:
        #     slicedInput.append(inputs[startId : startId + inputSize])
        #     startId += inputSize

        # Calculate input for each i, f, o
        # inputI = 0; inputF = 0; inputO = 0; inputG = 0
        # for idx, inputSize in enumerate(self.InputsSize):
        #     inputI += T.dot(slicedInput[idx], Uis[idx])
        #     inputF += T.dot(slicedInput[idx], Ufs[idx])
        #     inputO += T.dot(slicedInput[idx], Uos[idx])
        #     inputG += T.dot(slicedInput[idx], Ucs[idx])

        inputI = T.dot(inputs, Uis[0])
        inputF = T.dot(inputs, Ufs[0])
        inputO = T.dot(inputs, Uos[0])
        inputG = T.dot(inputs, Ucs[0])

        # Calculate to next layer
        i = T.nnet.sigmoid(inputI + T.dot(skm1, Wi) + bi)
        f = T.nnet.sigmoid(inputF + T.dot(skm1, Wf) + bf)
        o = T.nnet.sigmoid(inputO + T.dot(skm1, Wo) + bo)
        g = T.tanh(inputG + T.dot(skm1, Wc) + bc)

        self.C = ckm1 * f + g * i
        self.S = T.tanh(self.C) * o

        # Calculate output
        output = []
        for Wy, by in zip(Wys, bys):
            output.append(T.dot(self.S, Wy) + by)
        self.Output = output