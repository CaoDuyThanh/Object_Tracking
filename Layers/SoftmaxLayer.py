import theano.tensor as T
import numpy

class SoftmaxLayer():
    def __init__(self,
                 input):
        # Save information to its layer
        self.Input   = input

        self.Output = T.nnet.softmax(input)