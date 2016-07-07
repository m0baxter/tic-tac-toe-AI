
class Board(object):
    
    def __init__(self):
        self.squares = [ " ", " ", " ", " ", " ", " ", " ", " ", " " ]
        
    def showBoard(self):
        """Converts the board to a string for displaying purposes."""
        
        brd = "\n   |   |   \n" + \
              " " + self.squares[0] + " | " + self.squares[1] + " | " + self.squares[2] + " \n" + \
              "___|___|___\n" + \
              "   |   |   \n" + \
              " " + self.squares[3] + " | " + self.squares[4] + " | " + self.squares[5] + " \n" + \
              "___|___|___\n" + \
              "   |   |   \n" + \
              " " + self.squares[6] + " | " + self.squares[7] + " | " + self.squares[8] + " \n" + \
              "   |   |   \n"

        return brd
    
    def isBlank(self, n):
        """Checks if square n is blank."""
        
        return self.squares[n] == " "
    
    def markSquare(self, n, mrk):
        """Places marker mrk in square n."""
 
        self.squares[n] = mrk
        return
 
    def movesLeft(self):
        """Checks if any squares are still empty."""
 
        return " " in self.squares
 
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
 
    def convertBoard(self):
        """Converts a Board to a list of integers (Blank -> 0, X/O -> \\pm 1 )."""
 
        board = ""
 
        for m in self.squares:
           board += str(convertMarker(m)) + " "
 
        return board

    def intList(self):
        """returns the board as a list of integers (Blank -> 0, X/O -> \\pm 1 )."""

        board = []

        for m in self.squares:
            board.append(convertMarker(m))

        return board


def threeInARow(m1, m2, m3):
    """Checks if the the marks form a triple."""

    if " " in [m1, m2, m3]:
        return False

    else:
        return m1 == m2 and m1 == m3


def convertMarker(m):
    """Converts the marker to an integer: Blank -> 0, X/O -> \\pm 1."""
    
    if (m == " "):
        return 0
    elif (m == "X"):
        return 1
    elif (m == "O"):
        return -1
    else:
        raise ValueError("Bad marker descriptor.")

