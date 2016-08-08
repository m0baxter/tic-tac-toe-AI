
from board import *
import tttAI as ai
import numpy as np
import random as rnd
import time


def getReward(state, marker):
    """Given the state (a board) determine the reward."""

    if ( state.gameWon() ):
        return marker * 10
    else:
        return 0

def minMax(vals, flag):
    """Returns the max of vals if flag = -1 and the min if flag = 1."""

    if (flag == 1):
        return np.min(vals)
    elif (flag == -1):
        return np.max(vals)
    else:
        raise ValueError("Improper minMax flag.")


def updateQs(player, state, action, reward, newState):
    """Generates updated Q value for given state action pair."""

    y = player.getQs( np.array( state, ndmin = 2 ) )

    if (reward == 0):
        newQs = player.getQs( np.array( newState, ndmin = 2 ) )
        update = gamma * minMax(newQs, whoseTurn(state))

    else:
        update = reward

    y[0][action] = update

    return  y.reshape(9, )


def trainAI(player, nGames, gamma, epsilon, batchSize, memSize):
    """Trains the AI (player) to play tic tac toe using Q-learning algorithm over nGames games, with
       discount gamma, replay memory size memSize and batch update size of batchSize."""
       
    emptyBoard = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    
    xData = []
    yData = []
    memLoc = 0

    xWins = 0
    oWins = 0
    
    for age in range(nGames):
        
        rndPlayer = ai.RandomAI()
        board = Board()
        gameNotOver = True
        nTurns = 0

#        print  board.showBoard()
        
        while (gameNotOver):

            marker = (-1)**nTurns
            state = board.intList()

            #Determine which action to take:
            if ( state == emptyBoard or rnd.random() < epsilon):
                action = rndPlayer.takeTurn(board)

            else:
                action = player.takeTurn(board)

            #Take action (update board), get reward and new state:
            board.markSquare( action, marker )
            newState = board.intList()
            reward = getReward( board, marker)

            y =  updateQs( player, state, action, reward, newState )

            if ( len(xData) < memSize ):
                xData.append( state )
                yData.append( y )

                if ( len(xData) == memSize ):
                    xData = np.array(xData)
                    yData = np.array(yData)

            else:
                xData[memLoc] = np.array(state)
                yData[memLoc] = y

                indices = rnd.sample( xrange(memSize), batchSize )

                player.trainAI( xData[indices], yData[indices] )

            #Update tracking variables:
            gameNotOver = ( reward == 0 ) and ( board.movesLeft() )
            nTurns += 1
            memLoc = (memLoc + 1) % memSize

            if (reward == 10):
                xWins += 1
            elif (reward == -10):
                oWins += 1

#            print board.showBoard()

        #Decrease epsilon (lower randomness):
        if (epsilon > 0.1):
            epsilon -= (1.0/nGames)

        print "Finished Game:", age + 1, "X wins:", xWins, "O wins:", oWins

    return


if __name__ == "__main__":

    player = ai.NNAI()
    
    nGames    = 60000
    gamma     = 0.75
    epsilon   = 1.0
    batchSize = 500 #5000
    memSize   = 1000 #45000

    t1 = time.time()
    trainAI( player, nGames, gamma, epsilon, batchSize, memSize )
    t2 = time.time()

    print "\n\nTrained for {0} games in {1} seconds".format(nGames, t2 - t1)

    player.toFile( "player.h5" )

