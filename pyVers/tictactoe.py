
import board as brd
import neuralnet as nn
import tttAI as ai
import subprocess as sp


class TicTacToe(object):

    def __init__(self, path1, path2):

        #Hold neural net AIs for when they are needed:
        self.xAI = ai.NNAI( nn.NeuralNet.fromFile( path1 ) ) 
        self.oAI = ai.NNAI( nn.NeuralNet.fromFile( path2 ) ) 
        
        #Current game AIs:
        self.player1 = None
        self.player2 = None
        
        #Game board:
        self.board = brd.Board()

    def aiMenu(self, pNum):
        """The menu for setting AI."""

        sp.call("reset")

        try:
            choice = int( raw_input("Set player {0}:\n\n\t1 -> Human\n\t2 -> Neural network\n\t3 -> Random\n\n:".format(pNum)) )

        except ValueError:
            self.aiMenu(pNum)

        else:
            if (choice == 1 and pNum == 1):
                self.setAI(pNum, ai.HumanPlayer("X"))

            elif (choice == 1 and pNum == 2):
                self.setAI(pNum, ai.HumanPlayer("O"))

            elif (choice == 2 and pNum == 1):
                self.setAI(pNum, self.xAI)

            elif (choice == 2 and pNum == 2):
                self.setAI(pNum, self.oAI)

            elif (choice == 3):
                self.setAI(pNum, ai.RandomAI())

            else:
                self.aiMenu(pNum)

        return

    def takeTurn( self, mrk, ai ):
        """Player takes a turn."""
        
        sqr = ai.takeTurn(self.board)
        self.board.markSquare(sqr, mrk)

        return
 
    def playRound(self):
        """Play a round of tic-tac-toe."""
        
        self.board = brd.Board()
        print self.board.showBoard()
        
        while ( True ):
            
            self.takeTurn("X", self.player1)
            print self.board.showBoard()
            
            if ( self.board.gameWon() ):
                print "X wins."
                return

            elif ( not self.board.movesLeft() ):
                print "Meow."
                return
            
            self.takeTurn("O", self.player2)
            print self.board.showBoard()
            
            if ( self.board.gameWon() ):
                print "O wins."
                return

    def setAI(self, pNum, AI):
        """Sets player pNum to the AI."""

        if (pNum == 1):
            self.player1 = AI

        elif (pNum == 2):
            self.player2 = AI

        return

    def runGame(self):
        """Controls the menus between games."""

        play = "y"

        self.aiMenu(1)
        self.aiMenu(2)

        while (play == "y"):
            sp.call("reset")
            self.playRound()
            play = raw_input("Play again? [y/n/opt]: ")

            if (play == "opt"):
                self.aiMenu(1)
                self.aiMenu(2)

                play = "y"

        sp.call("reset")
        return


if __name__ == "__main__":

   #TicTacToe( "./genout/no-rand/p1-gen-118.mat", "./genout/no-rand/p2-gen-118.mat" ).runGame()
   TicTacToe( "./genout/p1-gen-752.mat", "./genout/p2-gen-752.mat" ).runGame()

