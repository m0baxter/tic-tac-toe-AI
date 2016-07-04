

import numpy as np


class NeuralNEt(object):

    def __init__(self, inputs, outputs, *hiddens):
        
        self.shape = [inputs] + list(hiddens) + [outputs]
        self.layers = len(self.shape)
        self.weights = self.__randInitialize()
    
    def __randInitialize(self):
        """Randomly initializes the weight matrices."""
        
        weights = []

        for i in xrange(self.layers - 1):
            Nin = self.shape[i]
            Nout = self.shape[i+1]

            eps = np.sqrt(6)/np.sqrt(Nin + Nout)
            weights.append( randMatrix((Nout, Nin + 1), eps)  )

        return weights

    def __unflatten(self, flat):
        """used by the cost function to unflatten weight matrices."""

        matrices = []
        start = 0

        for i in xrange(self.layers - 1):
            Nin = self.shape[i] +1
            Nout = self.shape[i+1]
            end = Nout * Nin + start

            arr = flat[start:end].reshape( (Nout, Nin) )
            matrices.append(arr)

            start = end

        return matrices
 

def randMatrix(size, eps):
    """Returns random matrix with shape = size whose values range in [-eps,eps]."""

    return 2*eps*np.random.random_sample(size) - eps


def sigmoid(z):
    """Returns the sigmoid function evaluated on z (z can be any numpy array or scalar)."""

    return 1/(1 + np.exp(-z))


if __name__ == "__main__":

    nn = NeuralNEt(9, 9, 5, 6)
    w = nn.weights

    flat = np.append(w[0].flatten(), w[1].flatten())
    flat = np.append(flat, w[2].flatten())

    recw = nn.unflatten(flat)

    print w[0] - recw[0]
    print w[1] - recw[1]
    print w[2] - recw[2]

