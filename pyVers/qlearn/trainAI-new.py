
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


def minMax(vals, flag):
    """Returns the max of vals if flag = -1 and the min if flag = 1."""

    if (flag == 1):
        return np.min(vals)

    elif (flag == -1):
        return np.max(vals)

    else:
        raise ValueError("Improper minMax flag.")


def updateQs(player, state, action, reward, newState, gamma, alpha):
    """Generates updated Q value for given state action pair."""

    y = player.getQs( np.array( state, ndmin = 2 ) )

    if (reward == 0):
        newQs = player.getQs( np.array( newState, ndmin = 2 ) )
        update = (1 - alpha) * y[0][action] + alpha * gamma * minMax(newQs, whoseTurn(state))

    else:
        update = reward

    y[0][action] = update
    y[0][ np.array(state) != 0 ] = 0

    return  y.reshape(9, )


def trainAI(player, sessions, nGames, gMin, gMax, alpha0, decay, epsilon, batchSize, gradSteps ):
    """Trains the AI (player) to play tic tac toe using Q-learning algorithm over nGames games, with
       discount gamma, replay memory size memSize and batch update size of batchSize."""

    emptyBoard = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    gamma = gMin
    alpha = alpha0
    epoch = 0
    avgLoss = 0.0

    for age in xrange(sessions):

        xData = []
        yData = []

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
                if ( state == emptyBoard or rnd.random() < epsilon ):
                    action = rndPlayer.takeTurn(board)

                else:
                    action = player.takeTurn(board)

                #Take action (update board), get reward and new state:
                board.markSquare( action, marker )
                newState = board.intList()
                reward = getReward( board, marker)

                y =  updateQs( player, state, action, reward, newState, gamma, alpha )

                xData.append( state )
                yData.append( y )

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

        xData = np.array(xData)
        yData = np.array(yData)

        losses = player.trainAI( xData, yData, batchSize, epoch + gradSteps, epoch = epoch )
        avgLoss = expMovingAverage( losses.history["loss"], 0.96, avg = avgLoss)

        print "Training session:", age + 1, "Loss:", avgLoss
        print

        #Decrease epsilon (lower randomness):
        if (epsilon > 0.1):
            epsilon -= (1.0/sessions)

        gamma = ( (gMax - gMin) * age )/(1.0 * sessions) + gMin
        alpha = alpha0 / (1.0 + decay * (age + 1))
        epoch += gradSteps

    return


def saveOnExit(player, path):
    """Function to execute on exit, saves player to file."""

    player.toFile( path )

    return


def expMovingAverage( vals, rate, avg = 0):

    if ( len(vals) == 0 ):
        return avg

    else:
        return expMovingAverage( vals[1:], rate, avg = rate * vals[0] + (1 - rate) * avg)


if __name__ == "__main__":

    player = None
    trainNew = True

    sessions  = 1000
    nGames    = 10000 #25000
    gMin      = 0.9
    gMax      = 0.9
    alpha     = 0.9
    decay     = 0.0
    epsilon   = 1.0
    batchSize = 128
    gradSteps = 50

    if ( trainNew ):
        player = ai.NNAI()
    else:
        player = ai.NNAI().fromFile( "player.h5" )
        gMin = gMax

    #Register function to save on exit (or kill):
    atexit.register( saveOnExit, player, "player-fail.h5" )

    print

    t1 = time.time()
    trainAI( player, sessions, nGames, gMin, gMax, alpha, decay, epsilon, batchSize, gradSteps )
    t2 = time.time()

    print "\n\nTrained for {0} games in {1} seconds".format(nGames * sessions, t2 - t1)

    saveOnExit(player, "player.h5")

