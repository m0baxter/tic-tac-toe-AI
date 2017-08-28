
import numpy as np


class Board(object):

    def __init__(self):
        self.squares = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    def showBoard(self):
        """Converts the board to a string for displaying purposes."""

        squares = self.stringList()

        brd = "\n   |   |   \n" + \
              " " + squares[0] + " | " + squares[1] + " | " + squares[2] + " \n" + \
              "___|___|___\n" + \
              "   |   |   \n" + \
              " " + squares[3] + " | " + squares[4] + " | " + squares[5] + " \n" + \
              "___|___|___\n" + \
              "   |   |   \n" + \
              " " + squares[6] + " | " + squares[7] + " | " + squares[8] + " \n" + \
              "   |   |   \n"

        return brd


    def isBlank(self, n):
        """Checks if square n is blank."""

        return self.squares[n] == 0

    def markSquare(self, n, mrk):
        """Places marker mrk in square n."""

        self.squares[n] = mrk
        return

    def movesLeft(self):
        """Checks if any squares are still empty."""

        return 0 in self.squares

    def gameWon(self):
        """Checks to see if the game has been won."""

        wins = [ threeInARow( self.squares[0], self.squares[1], self.squares[2] ),
                 threeInARow( self.squares[3], self.squares[4], self.squares[5] ),
                 threeInARow( self.squares[6], self.squares[7], self.squares[8] ),
                 threeInARow( self.squares[0], self.squares[3], self.squares[6] ),
                 threeInARow( self.squares[1], self.squares[4], self.squares[7] ),
                 threeInARow( self.squares[2], self.squares[5], self.squares[8] ),
                 threeInARow( self.squares[0], self.squares[4], self.squares[8] ),
                 threeInARow( self.squares[2], self.squares[4], self.squares[6] ) ]

        return any(wins)

    def stringList(self):
        """converts the board list to a list of X, O and blanks."""

        brd = []

        for m in self.squares:
            brd.append( convertMarker(m) )

        return brd

    def intList(self):
        """returns the board as a list of integers (Blank -> 0, X/O -> \\pm 1 )."""

        return self.squares[:]


def threeInARow(m1, m2, m3):
    """Checks if the the marks form a triple."""

    if 0 in [m1, m2, m3]:
        return False

    else:
        return m1 == m2 and m1 == m3


def convertMarker(m):
    """Converts the marker to an integer: Blank -> 0, X/O -> \\pm 1."""

    if (m == 0):
        return " "
    elif (m == 1):
        return "X"
    elif (m == -1):
        return "O"
    else:
        raise ValueError("Bad marker descriptor.")


def whoseTurn(board):
    """Returns marker for the player whose turn it is now."""

    if ( np.sum(board) == 1):
        return 1

    else:
        return -1

