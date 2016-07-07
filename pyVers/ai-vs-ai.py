

from board import *
from tttAI import *
import numpy as np


emptyBoard = "0 0 0 0 0 0 0 0 0 "

def takeTurn( mrk, brd ):
   """Ask player mrk for square to place their marker."""

   done = False
   boardStr = brd.convertBoard()
   ai = RandomAI()

   while (not done):

      sqr = ai.takeTurn(brd)

      if ( brd.isBlank(sqr) ):
         brd.markSquare(sqr, mrk)
         boardStr += str(sqr)
         done = True

   return boardStr


def playRound():
   """Play a round of tic-tac-toe."""

   brd = Board()

   p1Moves = ""
   p2Moves = ""

   while ( True ):

      #X moves:
      mv = takeTurn("X", brd)

      if (mv[:-1] == emptyBoard):
         mv = ""
      else:
         mv += "\n"

      p1Moves += mv

      #X wins:
      if ( brd.gameWon() ):
         return (p1Moves, "", np.array([1,0,0]))

      #cat's game:
      elif ( not brd.movesLeft() ):
         return (p1Moves, p2Moves, np.array([0,0,1]))
         #return ("", "", np.array([0,0,1]))

      p2Moves += takeTurn("O", brd) + "\n"

      #O wins:
      if ( brd.gameWon() ):
         return ("", p2Moves, np.array([0,1,0]))


def runGames( n ):
   """Plays a round. prompts for new game."""

   p1File = open("p1.txt", 'w')
   p2File = open("p2.txt", 'w')

   record = np.array([0,0,0])

   for i in xrange(n):
      p1Moves, p2Moves, res = playRound()

      p1File.write(p1Moves)
      p2File.write(p2Moves)

      record += res

   return record


if __name__ == "__main__":

   record = runGames(100)

   print "X: {0}, O: {1}, Tie: {2}".format(record[0], record[1], record[2])

