

import numpy as np


class NeuralNEt(object):

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
        Y = np.zeros(i (m, self.numLabels()) )

        for i in xrange(m):
            Y[i,:] = np.eye(nl)[ y[i] % nl, : ]

        return Y

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
        cost = -np.sum( np.log(As[-1])*Y + np.log(1 - As[-1] + 1.0e-15)*(1 - T))/m + reg

        #gradient:
        D3 = A3 - Y
        D2 = (D3.dot(Ws[1]) * np.append( np.ones((m,1)), sigGrad(z2)) )[:,1:end]

        Delta2 = D3.T.dot(A2)
        Delta1 = D2.T.dot(A1)

        mw1, _ = Ws[0].shape
        mw2, _ = Ws[1].shape

        grad1 =  ( Delta1 + l*np.append(np.zeros((mw1,1)), Ws[0][:,1:], 1) )/m
        grad2 =  ( Delta1 + l*np.append(np.zeros((mw2,1)), Ws[1][:,1:], 1) )/m

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

        return a3.argmax()


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


if __name__ == "__main__":

    nn = NeuralNEt(9, 3, 5)

    x = np.array([[1,2,3,4,5,6,7,8,9]])
    print nn.evaluate(x)

