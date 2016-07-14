
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
 
    def __init__(self, neuralnet):
        """Initializes an AI from a set of example moves."""

        self.neuralnet = neuralnet
 
    def takeTurn(self, board):
        """Takes one turn."""

        emptyBoard = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

        brdList = board.intList()

        if (brdList == emptyBoard):
            return rnd.randrange(0,9)

        else:
            rankedMvs = self.neuralnet.evaluate( np.array([brdList]) )
 
            for mv in rankedMvs:
                if ( board.isBlank(mv) ):
                    return mv

            raise RuntimeError("Could not make a move, the board was full.")

