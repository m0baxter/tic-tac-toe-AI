
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

        while (True):
            sqr = rnd.randrange(0,9)

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

