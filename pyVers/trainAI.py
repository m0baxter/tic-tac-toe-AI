
from board import *
import tttAI as ai
import neuralnet as nn
import numpy as np
import time
import threading


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


def writeMoves(moves, pNum):
    """Writes the moves to disk."""

    writeFile = open("p{0}.txt".format(pNum), 'w')
    writeFile.write(moves)
    writeFile.close()

    return


def generateData():
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

    print "\n\nRandom Vs. Random:"
    print "X: {0}, O: {1}, Tie: {2}\n\n".format(record[0], record[1], record[2])

    return


def trainAIs(nnO, nnX, itr):
    """Performs one training sted of the tic tac toe AI's."""

    dataX, yX = readin("p1.txt", int)
    dataO, yO = readin("p2.txt", int)

    t1 = threading.Thread( target = nnX.trainNetwork, args = (dataX, yX, l) )
    t2 = threading.Thread( target = nnO.trainNetwork, args = (dataO, yO, l) )

    start = time.time()

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    end = time.time()

    print "\n\nLesson {0} completed in: {1} (s)\n\n".format(itr, end - start)

    return


def learnTicTacToe(maxItr, l, inNodes, outNodes, *hiddenNodes):
    """Trains player to (first or second set by pNum) to play tic tac toe for at most maxItr lessons."""

    nnX = nn.NeuralNet(inNodes, outNodes, *hiddenNodes)
    nnO = nn.NeuralNet(inNodes, outNodes, *hiddenNodes)
    
    #Generate game data:
    generateData()

    for itr in xrange(1, maxItr + 1):

        trainAIs(nnO, nnX, itr)

        playerX = ai.NNAI(nnX)
        playerO = ai.NNAI(nnO)
    
        #X AI Vs random player:
        xMoves, _, xVsRand = playGames( playerX, ai.RandomAI(), nGames)
        writeMoves(xMoves, 1)
    
        #Random player Vs O AI:
        _, oMoves, oVsRand = playGames( ai.RandomAI(), playerO, nGames)
        writeMoves(oMoves, 2)
    
        #X AI Vs O AI:
        _, _, xVsO =  playGames( playerX, playerO, nGames)
    
        print "AI Vs. Random:"
        print "X: {0}, O: {1}, Tie: {2}\n\n".format( xVsRand[0], xVsRand[1], xVsRand[2])
    
        print "Random Vs. AI:"
        print "X: {0}, O: {1}, Tie: {2}\n\n".format( oVsRand[0], oVsRand[1], oVsRand[2])
    
        print "AI Vs. AI:"
        print "X: {0}, O: {1}, Tie: {2}\n\n".format( xVsO[0], xVsO[1], xVsO[2])

        nnX.toFile("p1-{0}".format(itr))
        nnO.toFile("p2-{0}".format(itr))

    return


def readin( path, DT = float ):
    
    data = np.loadtxt( path, delimiter = " ", dtype=DT)

    X = data[:,0:9]
    m,n = X.shape
    y = data[:,9].reshape((m,1))

    return (X, y)


if __name__ == "__main__":

    nGames = 12000
    maxItr = 3
    
    #Neural net parameters:
    l = 0.1
    inNodes = 9
    outNodes = 9
    hiddenNodes = (16, 16, 16)

    learnTicTacToe(maxItr, l, inNodes, outNodes, *hiddenNodes)

