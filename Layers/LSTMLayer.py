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
                 hkm1):
        # Save all information to its layer
        self.NetName     = net.NetName
        self.NumHidden   = net.LayerOpts['lstm_num_hidden']
        self.InputsSize  = net.LayerOpts['lstm_inputs_size']
        self.OutputsSize = net.LayerOpts['lstm_outputs_size']
        self.Params      = net.LayerOpts['lstm_params']

        if self.Params is None:
            # Parameters for list of input
            Wis = []; Wfs = []; Wcs = []; Wos = []            # Input Weight
            for idx, inputSize in enumerate(self.InputsSize):
                Wi = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 0.08, 1, '%s_Wi_%d' % (self.NetName, idx))
                Wf = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 0.08, 1, '%s_Wf_%d' % (self.NetName, idx))
                Wc = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 0.08, 1, '%s_Wc_%d' % (self.NetName, idx))
                Wo = CreateSharedParameter(net.NetOpts['rng'], (inputSize, self.NumHidden), 0.08, 1, '%s_Wo_%d' % (self.NetName, idx))

                Wis.append(Wi)
                Wfs.append(Wf)
                Wcs.append(Wc)
                Wos.append(Wo)

            # Init Ui | bi
            Ui = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 0.08, 1, '%s_Ui' % (self.NetName))
            bi = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0.08, 0, '%s_bi' % (self.NetName))

            # Init Uf | bf
            Uf = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 0.08, 1, '%s_Uf' % (self.NetName))
            bf = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0.08, 0, '%s_bf' % (self.NetName))

            # Init Uc | bc
            Uc = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 0.08, 1, '%s_Uc' % (self.NetName))
            bc = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0.08, 0, '%s_bc' % (self.NetName))

            # Init Uo | bo
            Uo = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 0.08, 1, '%s_Uo' % (self.NetName))
            bo = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0.08, 0, '%s_bo' % (self.NetName))

            # Parameters for list of output
            Wys = [];  bys = [];   # Output Weight | Bias
            for idx, outputSize in enumerate(self.OutputsSize):
                Wy = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, outputSize), 0.08, 1, '%s_Wy_%d' % (self.NetName, idx))
                by = CreateSharedParameter(net.NetOpts['rng'], (outputSize,), 0.08, 0, '%s_by_%d' % (self.NetName, idx))

                Wys.append(Wy)
                bys.append(by)

            self.Params = Wis + Wfs + Wcs + Wos + \
                          Wys + bys + \
                          [Ui,  Uf,   Uc,   Uo] + \
                          [bi,  bf,   bc,   bo]

        # Get all weight from param
        numInputs  = self.InputsSize.__len__()
        numOutputs = self.OutputsSize.__len__()
        Wis = self.Params[0 : numInputs]
        Wfs = self.Params[numInputs * 1 : numInputs * 2]
        Wcs = self.Params[numInputs * 2 : numInputs * 3]
        Wos = self.Params[numInputs * 3 : numInputs * 4]
        Wys = self.Params[numInputs * 4                  : numInputs * 4 + numOutputs * 1]
        bys = self.Params[numInputs * 4 + numOutputs * 1 : numInputs * 4 + numOutputs * 2]
        Ui  = self.Params[numInputs * 4 + numOutputs * 2]
        Uf  = self.Params[numInputs * 4 + numOutputs * 2 + 1]
        Uc  = self.Params[numInputs * 4 + numOutputs * 2 + 2]
        Uo  = self.Params[numInputs * 4 + numOutputs * 2 + 3]
        bi  = self.Params[numInputs * 4 + numOutputs * 2 + 4]
        bf  = self.Params[numInputs * 4 + numOutputs * 2 + 5]
        bc  = self.Params[numInputs * 4 + numOutputs * 2 + 6]
        bo  = self.Params[numInputs * 4 + numOutputs * 2 + 7]

        inputI = T.dot(inputs, Wis[0])
        inputF = T.dot(inputs, Wfs[0])
        inputO = T.dot(inputs, Wos[0])
        inputG = T.dot(inputs, Wcs[0])

        # Calculate to next layer
        i = T.nnet.sigmoid(inputI + T.dot(hkm1, Ui) + bi)
        f = T.nnet.sigmoid(inputF + T.dot(hkm1, Uf) + bf)
        o = T.nnet.sigmoid(inputO + T.dot(hkm1, Uo) + bo)
        g = T.tanh(inputG + T.dot(hkm1, Uc) + bc)

        self.C = ckm1 * f + g * i
        self.H = T.tanh(self.C) * o

        # Calculate output
        output = []
        for Wy, by in zip(Wys, bys):
            output.append(T.dot(self.H, Wy) + by)
        self.Output = output