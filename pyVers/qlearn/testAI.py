
from board import *
import tttAI as ai
import numpy as np
import random as rnd
import time
import atexit


def getReward(state, marker):
    """Given the state (a board) determine the reward."""

    if ( state.gameWon() ):
        return marker * 10
    else:
        return 0


def testAI( player, opponent, nGames ):
    """Trains the AI (player) to play tic tac toe using Q-learning algorithm over nGames games, with
       discount gamma, replay memory size memSize and batch update size of batchSize."""

    emptyBoard = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    xWins = 0
    oWins = 0

    for i in xrange(nGames):

        rndPlayer = ai.RandomAI()
        board = Board()
        gameNotOver = True
        nTurns = 0

        while (gameNotOver):

            marker = (-1)**nTurns
            state = board.intList()

            #Determine which action to take:
            if ( state == emptyBoard or opponent == marker ):
                action = rndPlayer.takeTurn(board)

            else:
                action = player.takeTurn(board)

            #Take action (update board), get reward and new state:
            board.markSquare( action, marker )
            reward = getReward( board, marker)

            #Update tracking variables:
            gameNotOver = ( reward == 0 ) and ( board.movesLeft() )
            nTurns += 1

            if (reward == 10):
                xWins += 1
            elif (reward == -10):
                oWins += 1

    print "Games played:", i + 1, \
          "X wins:", xWins, \
          "O wins:", oWins, \
          "Ties:", (i + 1) - (xWins + oWins)

    return


if __name__ == "__main__":

    nGames = 10000
    player = ai.NNAI().fromFile( "player.h5" )

    print "AI vs Random:"
    testAI( player, -1, nGames )
    print
    print "Random vs AI:"
    testAI( player, 1, nGames )
    print
    print "AI vs AI:"
    testAI( player, 0, nGames )
    print

