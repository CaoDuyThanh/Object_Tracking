import theano
import theano.tensor as T

class AdamGDUpdate():
    def __init__(self,
                 net,
                 params,
                 grads):
        # Save all information to its layer
        self.Beta1        = net.UpdateOpts['adam_beta1']
        self.Beta2        = net.UpdateOpts['adam_beta1']
        self.Delta        = net.UpdateOpts['adam_delta']
        self.LearningRate = net.NetOpts['learning_rate']


        updates = []
        for (param, grad) in zip(params, grads):
            mt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            vt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)

            newMt = self.Beta1 * mt + (1 - self.Beta1) * grad
            newVt = self.Beta2 * vt + (1 - self.Beta2) * T.sqr(grad)

            tempMt = newMt / (1 - self.Beta1)
            tempVt = newVt / (1 - self.Beta2)

            step = - self.LearningRate * tempMt / (T.sqrt(tempVt) + self.Delta)
            updates.append((mt, newMt))
            updates.append((vt, newVt))
            updates.append((param, param + step))

        self.Updates = updates

