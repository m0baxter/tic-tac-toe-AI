
import neuralnet as nn
import tttAI as ai
import board as brd
import numpy as np
import random as rnd
import time


class Player(object):

    def __init__(self, ai):

        self.ai = ai
        self.loses = 0
        self.ties  = 0
        self.wins  = 0
        self.games = 0

    def won(self):
        """Increases the number of wins by one."""

        self.wins += 1
        return

    def lose(self):
        """Increases the number of loses by one."""

        self.loses += 1
        return

    def tie(self):
        """Increases the number of ties by one"""

        self.ties += 1
        return

    def getRecord(self):
        """Returns (Win, Lose, Tie)."""

        return (self.wins, self.loses, self.ties)

    def getAI(self):
        """Returns the players AI."""

        return self.ai

    def fitness(self):
        """calculates the fitness of the player."""

        rec = self.getRecord()
        nPlayed = sum( rec )

        return ( 1 - float(rec[1]) / nPlayed )


class Tournament(object):

    def __init__(self, breedPop, pCross, pMut):

        self.breedPop = breedPop
        self.pCross = pCross
        self.pMut = pMut

        self.xPlayers = []
        self.oPlayers = []

    def genPlayers( self, nPlayers ):
        """Generates nPlayers X and O AI's."""

        for i in xrange(nPlayers):

            nnX = nn.NeuralNet.fromFile("p1-1.mat")
            nnO = nn.NeuralNet.fromFile("p2-1.mat")

            nnX.laplaceWeights(0.0, 1.0)
            nnO.laplaceWeights(0.0, 1.0)

#            for w in nnX.weights:
#                s = w.shape
#                w += np.random.laplace(0, 1, s)

#            for w in nnO.weights:
#                s = w.shape
#                w += np.random.laplace(0, 1, s)

            self.xPlayers.append( Player( ai.NNAI(nnX) ) )
            self.oPlayers.append( Player( ai.NNAI(nnO) ) )

        return

    def playTournament( self, nGames ):
        """Each X-player plays each O-player and random player nGames times."""

        rndPlayer = Player( ai.RandomAI() )
        
        #play against other players:
        for p1 in self.xPlayers:
            for p2 in self.oPlayers:
                playGames( p1, p2, nGames )

        #Play against randoms to see more boards:
        for i in xrange( len(self.xPlayers) ):
            playGames( self.xPlayers[i], rndPlayer, nGames )
            playGames( rndPlayer, self.xPlayers[i], nGames )

        return

    def maxFitness(self):
        """Returns the maximum fitness in the current population."""

        maxFitX = np.max( [ p.fitness() for p in self.xPlayers ] )
        maxFitO = np.max( [ p.fitness() for p in self.oPlayers ] )

        return ( maxFitX, maxFitO )

    def selectBreeders(self):
        """Selects the X- and O-players to enter the breeding pool."""

        maxFitX , maxFitO = self.maxFitness()

        xPop = [ (p, p.fitness()/maxFitX) for p in self.xPlayers]
        oPop = [ (p, p.fitness()/maxFitO) for p in self.oPlayers]

        xPop.sort(key = lambda a : a[1], reverse = True)
        oPop.sort(key = lambda a : a[1], reverse = True)

        xCandidates = []
        oCandidates = []

        for i in xrange( len(xPop) ):

            if ( rnd.random() <= xPop[i][1] ):
                xCandidates.append(xPop[i][0])
                
            if ( rnd.random() <= oPop[i][1]  ):
                oCandidates.append(oPop[i][0])

        xSelected = [ xCandidates.pop(0) ]
        oSelected = [ oCandidates.pop(0) ]

        rnd.shuffle(xCandidates)
        rnd.shuffle(oCandidates)

        return ( xSelected + xCandidates[:self.breedPop-1], oSelected + oCandidates[:self.breedPop-1] )

    def breed(self, stock):
        """Breeds a new generation of players."""

        nextGen = []

        for p1 in stock:
            nn1 = p1.getAI().neuralnet
            shape = nn1.shape
            layers = nn1.layers

            W1 = nn1.weights

            for p2 in stock:

                nn2 = p1.getAI().neuralnet

                W2 = nn2.weights

                Wc1 = []
                Wc2 = []

                for i in xrange( len(W1) ):

                    c1, c2 = uniCross(W1[i], W2[i], self.pCross)

                    c1 = mutate(c1, self.pMut)
                    c2 = mutate(c2, self.pMut)

                    Wc1.append(c1)
                    Wc2.append(c2)

                nnC1 = nn.NeuralNet(1,1)
                nnC2 = nn.NeuralNet(1,1)

                nnC1.shape = shape
                nnC2.shape = shape

                nnC1.layers = layers
                nnC2.layers = layers

                nnC1.weights = Wc1
                nnC2.weights = Wc2

                nextGen += [ Player( ai.NNAI(nnC1) ), Player( ai.NNAI(nnC2) ) ]

        return nextGen + stock

    def genNewPlayers(self):

        stockX, stockO = self.selectBreeders()

        nextGenX = self.breed( stockX )
        nextGenO = self.breed( stockO )

        self.xPlayers = nextGenX
        self.oPlayers = nextGenO

        return

    def saveBest(self):
        """Saves the weights of the best X- and O-players to disk."""

        xPop = [ (p, p.fitness()) for p in self.xPlayers]
        oPop = [ (p, p.fitness()) for p in self.oPlayers]

        xPop.sort( key = lambda a : a[1], reverse = True )
        oPop.sort( key = lambda a : a[1], reverse = True )

        bestX = xPop.pop(0)[0]
        bestO = oPop.pop(0)[0]

        nnX = bestX.getAI().neuralnet
        nnO = bestO.getAI().neuralnet

        nnX.toFile("p1-gen")
        nnO.toFile("p2-gen")

        return


def uniCross(a, b, pCross):
    """Performs unform cross over at ratio pCross, produces two offsprings."""
    
    s = a.shape
    childA = np.copy(a)
    childB = np.copy(b)

    mask = np.random.random_sample(s) <= pCross

    childA[mask] = b[mask]
    childB[mask] = a[mask]

    return (childA, childB)


def mutate(a, pMut):
    """Mutates alleles of a with probability pMut."""

    s = a.shape

    mask = np.random.random_sample(s) <= pMut

    a[mask] += np.random.laplace(0,1,s)[mask]

    return a


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


def playRound(p1, p2):
    """Play a round of tic-tac-toe."""
    
    b = brd.Board()
    p1AI = p1.getAI()
    p2AI = p2.getAI()
    
    while ( True ):
        
        #X moves:
        mv = takeTurn("X", b, p1AI)
        
        #X wins:
        if ( b.gameWon() ):
            p1.won()
            p2.lose()
            return
        
        #cat's game:
        elif ( not b.movesLeft() ):
            p1.tie()
            p2.tie()
            return

        takeTurn("O", b, p2AI)
        
        #O wins:
        if ( b.gameWon() ):
            p1.lose()
            p2.won()
            return


def playGames( player1, player2, n ):
    """Plays a round. prompts for new game."""

    for i in xrange(n):
        playRound(player1, player2)

    return


if __name__ == "__main__":

    breedPop = 7 #105 in every generation after the first. (as close to 100 as I can get)
    pCross = 0.5
    pMut = 0.03
    N0 = 105
    numGames = 20
    numGens = 120

    t = Tournament( breedPop, pCross, pMut)
    t.genPlayers( N0 )

    start = time.time()

    for i in range(numGens):

        print
        print "Generation {0}".format(i)
        print

        t1 = time.time()

        t.playTournament( numGames )

        print t.maxFitness()
        
        if ( i == numGens - 1):
            t.saveBest()

        t.genNewPlayers()

        t2 = time.time()

        print "Length of generation (s): ", t2 - t1

    end = time.time()

    print
    print
    print "Total time (s): ", end - start
    print

