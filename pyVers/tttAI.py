
import abc
import random as rnd


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

