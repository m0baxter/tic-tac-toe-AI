
from board import *
import tttAI as ai
import neuralnet as nn
import numpy as np
import time
import threading
import sys


emptyBoard = "0 0 0 0 0 0 0 0 0 "
nGames = None

def takeTurn( mrk, brd, ai ):
   """Ask player mrk for square to place their marker."""

   done = False
   boardStr = brd.convertBoard()

   while (not done):

      sqr = ai.takeTurn(brd)

      if ( brd.isBlank(sqr) ):
         brd.markSquare(sqr, mrk)
         boardStr += str(sqr)
         done = True

   return boardStr


def playRound(p1AI, p2AI):
   """Play a round of tic-tac-toe."""

   brd = Board()

   p1Moves = ""
   p2Moves = ""

   while ( True ):

      #X moves:
      mv = takeTurn("X", brd, p1AI)

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

      p2Moves += takeTurn("O", brd, p2AI) + "\n"

      #O wins:
      if ( brd.gameWon() ):
         return ("", p2Moves, np.array([0,1,0]))


def playGames( player1, player2, n ):
    """Plays a round. prompts for new game."""
    
    p1Moves = ""
    p2Moves = ""
    record = np.array([0,0,0])
    
    for i in xrange(n):

        mvs1, mvs2, res = playRound(player1, player2)

        p1Moves += mvs1 
        p2Moves += mvs2
        record += res

    return (p1Moves, p2Moves, record)

def initializeLearning():
    """Runs lesson zero where random players generate game data."""

    p1File = open("p1.txt", 'w')
    p2File = open("p2.txt", 'w')

    p1AI = ai.RandomAI()
    p2AI = ai.RandomAI()

    p1Moves, p2Moves, record = playGames( p1AI, p2AI, nGames)

    p1File.write(p1Moves)
    p2File.write(p2Moves)

    p1File.close()
    p2File.close()

    print "X: {0}, O: {1}, Tie: {2}".format(record[0], record[1], record[2])

    return

def learnTicTacToe(pNum, maxItr):
    """Trains player to (first or second set by pNum) to play tic tac toe for at most maxItr lessons."""

    playerStr = "p" + str(pNum)
    path = playerStr + ".txt"

    logFile = open( playerStr + ".log", 'w')

    nnPlayer = nn.NeuralNet(9, 9, 12)

    itr = 1
    start = time.time()

    while ( (itr <= maxItr) ): #(record good enough)
        
        X, y = readin(path, int)

        #Train network
        t1 = time.time()
        nnPlayer.trainNetwork(X, y, 0.1)
        t2 = time.time()

        logFile.write( "Lesson: {0}  Training Time: {1}\n".format( itr, t2 - t1 ) )

        if (pNum == 1):
            p1AI = ai.NNAI(nnPlayer)
            p2AI = ai.RandomAI()

        elif (pNum == 2):
            p1AI = ai.RandomAI()
            p2AI = ai.NNAI(nnPlayer)

        else:
            raise ValueError("pNum makes no sense")
        
        results = playGames( p1AI, p2AI, nGames)

        #Save win/tie moves to disk:
        mvsFile = open(path, 'w')
        mvsFile.write( results[pNum - 1] )
        mvsFile.close()

        logFile.write( "X: {0}, O: {1}, Tie: {2}\n\n".format( results[2][0],
                                                              results[2][1],
                                                              results[2][2]) )
        print pNum, itr
        itr += 1

    end = time.time()

    logFile.write( "\nRan {0} of {1} iterations  Training Time: {2}".format( itr, maxItr, end - start ) )

    #Save neural nework to file:
    nnPlayer.toFile(playerStr)

    return


def readin( path, DT = float ):
    
    data = np.loadtxt( path, delimiter = " ", dtype=DT)

    X = data[:, 0:9]
    m,n = X.shape
    y = data[:,9].reshape((m,1))

    return (X, y)


if __name__ == "__main__":

    try:
        nGames = int(sys.argv[1])

    except ValueError:
        print "Invalid nGames, setting nGames to 7000:"
        nGames = 7000

    initializeLearning()

    t1 = threading.Thread( target = learnTicTacToe, args = (1, 2) )
    t2 = threading.Thread( target = learnTicTacToe, args = (2, 2) )

    t1.start()
    t2.start()

    t1.join()
    t2.join()

