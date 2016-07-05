
import abc
import random as rnd
import numpy as np


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

   def takeTurn(self, board):
      """Takes one turn."""
      
      #fill in later
      pass


def readin( path, DT = float ):

    data = np.loadtxt( path, delimiter = " ", dtype=DT)
    X = data[:, 0:9]
    m,n = X.shape
    print m,n
    y = data[:,9].reshape((m,1))

    return (X, y)

