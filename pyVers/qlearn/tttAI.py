
import abc
import random as rnd
import numpy as np
import board as brd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import RMSprop


class TicTacToeAI(object):
 
    __metaclass__ = abc.ABCMeta
 
    @abc.abstractmethod
    def takeTurn(self, board):
        """Takes one turn."""
        pass


class RandomAI(TicTacToeAI):
 
    def takeTurn(self, board):
        """Randomly places a marker in a square."""

        moves = range(0,9)
        rnd.shuffle(moves)

        for sqr in moves:
            if board.isBlank(sqr):
                return sqr
 

class HumanPlayer(TicTacToeAI):

    def __init__(self, mrk):
        self.mrk = mrk
 
    def takeTurn(self, board):
        """Takes one turn."""

        while (True):
            try:
                sqr = int( raw_input(self.mrk +"'s turn [0-8]: ") )
                
            except ValueError:
                continue

            else:
                if ( 0 <= sqr and sqr < 9 and board.isBlank(sqr) ):
                    return sqr


class NNAI(TicTacToeAI):
 
    def __init__(self):
        """Initializes an AI from a set of example moves."""

        self.nn = Sequential()
        self.nn.add( Dense( 100, init='lecun_uniform', input_shape=(9,) ) )
        #self.nn.add( Activation('relu') )
        self.nn.add( PReLU() )
        #self.nn.add(Dropout(0.2))

        self.nn.add(Dense(100, init='lecun_uniform'))
        #self.nn.add(Activation('relu'))
        self.nn.add( PReLU() )
        #self.nn.add(Dropout(0.2))

        self.nn.add(Dense(9, init='lecun_uniform'))
        self.nn.add(Activation('linear'))

        self.nn.compile( loss = 'mse', optimizer = RMSprop() )

    @classmethod
    def fromFile(cls, path):
        """loads a neutal network from a file."""

        ai = NNAI()
        ai.nn.load_weights(path)

        return ai

    def toFile(self, path):
        """Saves the neural network to a file."""

        self.nn.save_weights(path, overwrite=True)

        return

    def trainAI(self, X, y):
        """Trains the neural network on the data X, Y."""

        m = len(X)
        self.nn.fit( X, y, batch_size = m, nb_epoch = 1, verbose = 1 )

        return

    def getQs(self, state):
        """Returns a numpy array of the Q(s,a) foar all possible actions a."""
        
        return self.nn.predict( state, batch_size=1)

    def takeTurn(self, board):
        """Takes one turn."""

        emptyBoard = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

        brdList = board.intList()

        if (brdList == emptyBoard):
            return rnd.randrange(0,9)

        else:
            rankedMvs = indexDecending( self.getQs( np.array( brdList, ndmin = 2 ) ) )

            if ( brd.whoseTurn(brdList == -1) ):
                rankedMvs.reverse()
 
            for mv in rankedMvs:
                if ( board.isBlank(mv) ):
                    return mv

            raise RuntimeError("Could not make a move, the board was full.")

def indexDecending( arr ):
    """Given the output of getQs return a list of decreaseing q values."""
    
    lst = list(arr[0])
    
    indices = []
    
    for i in xrange( len(lst) ):
        j = np.argmax(lst)

        indices.append(j)
        lst[j] = None

    return indices

