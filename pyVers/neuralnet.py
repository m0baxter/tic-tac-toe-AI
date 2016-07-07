
import numpy as np
from scipy.optimize import minimize


class NeuralNet(object):

    def __init__(self, inputs, outputs, *hiddens):
        
        self.shape = [inputs] + list(hiddens) + [outputs]
        self.layers = len(self.shape)
        self.weights = self.__randInitialize()

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

    def cost(self, theta, X, Y, l):
        m, n = X.shape

        Ws = self.__unflatten(theta)

        A1 = np.append( np.ones((m,1)), X, 1)
        Z2 = A1.dot(Ws[0].T)
        mz, nz = Z2.shape
        A2 = np.append( np.ones((mz,1)), sigmoid(Z2), 1)
        Z3 = A2.dot(Ws[1].T)
        A3 = sigmoid(Z3)

        #compute cost:
        reg = l*np.sum( [ np.sum( w[:,1:]**2 ) for w in Ws ] )/(2.0 * m)
        cost = -np.sum( np.log(A3)*Y + np.log(1 - A3 + 1.0e-15)*(1 - Y))/m + reg

        return cost


    def __nnCost(self, theta, X, Y, l):
        """Computes the cost and its gradient for the neural network on data X, Y
           with regularizer l."""

        m, n = X.shape

        Ws = self.__unflatten(theta)

        A1 = np.append( np.ones((m,1)), X, 1)
        Z2 = A1.dot(Ws[0].T)
        mz, nz = Z2.shape
        A2 = np.append( np.ones((mz,1)), sigmoid(Z2), 1)
        Z3 = A2.dot(Ws[1].T)
        A3 = sigmoid(Z3)

        #compute cost:
        reg = l*np.sum( [ np.sum( w[:,1:]**2 ) for w in Ws ] )/(2.0 * m)
        cost = -np.sum( np.log(A3)*Y + np.log(1 - A3 + 1.0e-15)*(1 - Y))/m + reg

        #gradient:
        D3 = A3 - Y
        D2 = ( D3.dot(Ws[1]) * np.append(np.ones((m,1)), sigGrad(Z2), 1) )[:,1:]

        Delta2 = D3.T.dot(A2)
        Delta1 = D2.T.dot(A1)

        mw1, _ = Ws[0].shape
        mw2, _ = Ws[1].shape

        grad1 =  ( Delta1 + l*np.append(np.zeros((mw1,1)), Ws[0][:,1:], 1) )/m
        grad2 =  ( Delta2 + l*np.append(np.zeros((mw2,1)), Ws[1][:,1:], 1) )/m

#        GRAD = np.append(grad1.flatten(), grad2.flatten())
#        f = lambda x : self.cost(x, X,Y,l)
#        NGRAD = numGrad(f,theta)
#
#        print np.linalg.norm(GRAD - NGRAD)/np.linalg.norm(GRAD + NGRAD)

        return ( np.asscalar(cost), np.append(grad1.flatten(), grad2.flatten()) )

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

        print a3

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


def numGrad(f, x):
    """Computes numerical gradient of f at x."""

    numgrad = np.zeros(len(x))
    perturb = np.zeros(len(x))
    e = 1.0e-4

    for p in xrange(len(x)):
        perturb[p] = e

        fp = f(x + perturb)
        fm = f(x - perturb)

        numgrad[p] = (fp - fm)/(2*e)

        perturb[p] = 0.0

    return numgrad




def readin( path, DT = float ):
    
    data = np.loadtxt( path, delimiter = " ", dtype=DT)

    X = data[:, 0:9]
    m,n = X.shape
    y = data[:,9].reshape((m,1))

    return (X, y)


if __name__ == "__main__":

    import time

    nn = NeuralNet(9, 9, 8)

    X, y = readin("p1.txt", int)
    X.astype(float)

    print
    t1 = time.time()
    nn.trainNetwork(X, y, 0.0)
    t2 = time.time()
    print
    print "training time (s): ", t2 - t1
    print

    x1 = np.array([ [1, -1, 0,
                     1, -1, 0,
                     0, 0, 0] ])
    x2 = np.array([ [0, 0, 1,
                     0, 0, -1,
                     1, 0, -1] ])
    x3 = np.array([ [1, -1, 0,
                     -1, 1, 0,
                     0, 0, 0] ])
    x4 = np.array([ [-1, 0, -1,
                     0, 0, 0,
                     1, 0, 1] ])
    x5 = np.array([ [0, 0, 0,
                     0, -1, -1,
                     0, 1, 1] ])
    x6 = np.array([ [0, 1, -1,
                     0, 1, 0,
                     0, 0, -1] ])
    x7 = np.array([ [0, 0, 0,
                     -1, 1, 0,
                     1, 0, -1] ])
    x8 = np.array([ [1, 0, -1,
                     0, 0, -1,
                     0, 0, 1] ])
    x9 = np.array([ [0, 1, 1,
                     0, 0, -1,
                     -1, 0, 0] ])
    x10 = np.array([ [0, 0, 1,
                     0, -1, 0,
                     0, -1, 1] ])

    print "test 1:", nn.evaluate(x1), "6 (7)"
    print "test 2:", nn.evaluate(x2), 4
    print "test 3:", nn.evaluate(x3), 8
    print "test 4:", nn.evaluate(x4), "7 (1)"
    print "test 5:", nn.evaluate(x5), "6 (3)"
    print "test 6:", nn.evaluate(x6), "7 (5)"
    print "test 7:", nn.evaluate(x7), 2
    print "test 8:", nn.evaluate(x7), 4
    print "test 9:", nn.evaluate(x7), 0
    print "test 10:", nn.evaluate(x7), "5 (1)"
    print

#    print nn.weights[0]
#    print "\n"
#    print nn.weights[1]
#    print

