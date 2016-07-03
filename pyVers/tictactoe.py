
from board import *


def takeTurn( mrk, brd ):
   """Ask player mrk for square to place their marker."""

   done = False

   while (not done):
      try:
         sqr = int( raw_input(mrk +"'s turn [0-8]: ") )

      except ValueError:
         continue

      else:
         if ( 0 <= sqr and sqr < 9 and brd.isBlank(sqr) ):
            brd.markSquare(sqr, mrk)
            done = True

         else:
            continue
   return


def playRound():
   """Play a round of tic-tac-toe."""

   brd = Board()
   print brd.showBoard()

   while ( True ):

      takeTurn("X", brd)
      print brd.showBoard()

      if ( brd.gameWon() ):
         print "X wins."
         return

      elif ( not brd.movesLeft() ):
         print "Meow."
         return

      takeTurn("O", brd)
      print brd.showBoard()

      if ( brd.gameWon() ):
         print "O wins."
         return


def runGame():
   """Plays a round. prompts for new game."""

   play = "y"

   while ( play == "y"):
      playRound()
      play = raw_input("Play again? [y/n]: ")
		

if __name__ == "__main__":

   runGame()

