
import abc
import random as rnd
import numpy as np
import neuralnet as nn


class TicTacToeAI(object):
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def takeTurn(self, board):
        """Takes one turn."""
        pass


class RandomAI(TicTacToeAI):
    
    def takeTurn(self, board):
        """Randomly places a marker in a square."""
        
        return rnd.randrange(0,9)


class HumanPlayer(TicTacToeAI):
    
    def takeTurn(self, board):
        """Takes one turn."""

        #fill in later
        pass


class NNAI(TicTacToeAI):
    
    def __init__(self, path):
        """Initializes an AI from a set of example moves."""

        self.neuralnet = nn.NeuralNEt(9, 9, 8)

        X, y = readin(path, int)
        self.neuralnet.trainNetwork(X, y, 0.1)
        
    def takeTurn(self, board):
        """Takes one turn."""
        
        brdVect = np.array([board.intList()])
        rankedMvs = self.neuralnet.evaluate(brdVect)

        for mv in rankedMvs:
            if ( board.isBlank(mv) ):
                return mv

        raise RuntimeError("Could not make a move, the board was full.")


def readin( path, DT = float ):
    
    data = np.loadtxt( path, delimiter = " ", dtype=DT)

    X = data[:, 0:9]
    m,n = X.shape
    print m,n
    y = data[:,9].reshape((m,1))

    return (X, y)

