
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize


class NeuralNet(object):

    def __init__(self, inputs, outputs, *hiddens):
        
        self.shape = [inputs] + list(hiddens) + [outputs]
        self.layers = len(self.shape)
        self.weights = self.__randInitialize()

    def toFile(self, fileName):
        """Saves the NeuralNet to filename.mat."""

        nnDict = dict( shape = self.shape, weights=self.weights)
        sio.savemat(fileName, nnDict, True)

        return

    @classmethod
    def fromFile(cls, path):
        """Creates a NeuralNet from a file."""

        cls = NeuralNet(1,1,1)

        data = sio.loadmat(path)

        cls.shape = list(data['shape'][0])
        cls.layers = len(cls.shape)
        cls.weights = list(data['weights'][0])

        return cls

    def numLabels(self):
        """Returns the number labels in the output of the network."""

        return self.shape[-1]
    
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

    def __labelMatrix(self, y):
        """converts the label data into a matrix of zeroes and ones."""

        m, n = y.shape
        nl = self.shape[-1]
        Y = np.zeros( (m, self.numLabels()) )

        for i in xrange(m):
            Y[i,:] = np.eye(nl)[ y[i], : ] #% nl

        return Y

    def __nnCost(self, theta, X, Y, l):
        """Computes the cost and its gradient for the neural network on data X, Y
           with regularizer l."""

        m, n = X.shape

        Ws = self.__unflatten(theta)
        Grads = [ np.zeros(w.shape) for w in Ws ]
        As = [ np.append( np.ones((m,1)), X, 1) ]
        Zs = []

        for l in xrange(self.layers - 2):
            #compute Z:
            z = As[l].dot(Ws[l].T)
            Zs.append(z)
            
            #Compute activation (A):
            mz, nz = z.shape
            a = np.append( np.ones((mz,1)), sigmoid(z), 1)
            As.append(a)

        #activation of output:
        Zs.append( As[-1].dot(Ws[-1].T) )
        As.append( sigmoid(Zs[-1]) )

        #compute cost:
        reg = l*np.sum( [ np.sum( w[:,1:]**2 ) for w in Ws ] )/(2.0 * m)
        cost = -np.sum( np.log(As[-1])*Y + np.log(1 - As[-1] + 1.0e-15)*(1 - Y))/m + reg

        #gradient:
        d = As[-1] - Y
        Delta = d.T.dot(As[-2])
        mw, _ = Ws[-1].shape
        Grads[-1] =  ( Delta + l*np.append(np.zeros((mw,1)), Ws[-1][:,1:], 1) )/m

        for i in xrange(2, self.layers):
            z = sigGrad(Zs[-i])
            mz, _ = z.shape

            d = ( d.dot(Ws[-i+1]) * np.append(np.ones((mz,1)), z, 1) )[:,1:]
            Delta = d.T.dot(As[-i - 1])

            mw, _ = Ws[-i].shape
            Grads[-i] = ( Delta + l*np.append(np.zeros((mw,1)), Ws[-i][:,1:], 1) )/m

        GRAD = np.concatenate( [g.flatten() for g in Grads] )

        return ( np.asscalar(cost), GRAD )

    def trainNetwork(self, X, y, l):
        """Trains the neural network from the data X with labels y and regularizer l."""
        
        Y = self.__labelMatrix(y)
        theta = np.concatenate([w.flatten() for w in self.weights])

        res = minimize( self.__nnCost, theta, args=(X,Y,l), method='CG',
                        jac =True, options={'disp': True})

        self.weights = self.__unflatten(res.x)

        return

    def evaluate(self, x):
        """Evaluates the neural network on a single example x, a 1xn vector."""

        a1 = np.append( np.ones((1,1)), x, 1)
        z2 = a1.dot(self.weights[0].T)
        mz, nz = z2.shape
        a2 = np.append( np.ones((mz,1)), sigmoid(z2), 1)
        z3 = a2.dot(self.weights[1].T)
        a3 = sigmoid(z3)

        #print a3

        return self.__indexDecending(a3)
    
    def __indexDecending( self, arr ):
        """Given the output of NeuralNet.evaluate return a list of decreasing certainty
           of classification."""

        lst = list(arr[0])
        
        indices = []
        
        for i in xrange( len(lst) ):
            j = np.argmax(lst)

            indices.append(j)
            lst[j] = None

        return indices


def randMatrix(size, eps):
    """Returns random matrix with shape = size whose values range in [-eps,eps]."""

    return 2*eps*np.random.random_sample(size) - eps


def sigmoid(z):
    """Returns the sigmoid function evaluated on z (z can be any numpy array or scalar)."""

    return 1/(1 + np.exp(-z))


def sigGrad(z):
    """Returns the gradient of the sigmoid at z (saclar or numpy array)."""

    s = sigmoid(z)
    return s * (1 - s)


def readin( path, DT = float ):
    
    data = np.loadtxt( path, delimiter = " ", dtype=DT)

    X = data[:, 0:9]
    m,n = X.shape
    y = data[:,9].reshape((m,1))

    return (X, y)

