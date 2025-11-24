# Utilities 
import chess
from chess import Board, Move

# Board coordinate annotation
ROWS = 'abcdefgh'
COLS = '12345678'

# Score for per pieces
pieceScore = {
    "K": 0,
    "Q": 90,
    "R": 50,
    "B": 35,
    "N": 30,
    "P": 10 
}

# Position score matrices for non-king pieces
knightScore = [[1, 1, 1, 1, 1, 1, 1, 1],
               [1, 2, 2, 2, 2, 2, 2, 1],
               [1, 2, 3, 3, 3, 3, 2, 1],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [1, 2, 3, 3, 3, 3, 2, 1],
               [1, 2, 2, 2, 2, 2, 2, 1],
               [1, 1, 1, 1, 1, 1, 1, 1]]

bishopScore = [[4, 3, 2, 1, 1, 2, 3, 4],
               [3, 4, 3, 2, 2, 3, 4, 3],
               [2, 3, 4, 3, 3, 4, 3, 2],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [2, 3, 4, 3, 3, 4, 3, 2],
               [3, 4, 3, 2, 2, 3, 4, 3],
               [4, 3, 2, 1, 1, 2, 3, 4]]

queenScore = [[1, 1, 1, 3, 1, 1, 1, 1],
              [1, 2, 3, 3, 3, 1, 1, 1],
              [1, 4, 3, 3, 3, 4, 2, 1],
              [1, 2, 3, 3, 3, 2, 2, 1],
              [1, 2, 3, 3, 3, 2, 2, 1],
              [1, 4, 3, 3, 3, 4, 2, 1],
              [1, 2, 3, 3, 3, 1, 1, 1],
              [1, 1, 1, 3, 1, 1, 1, 1]]

rookScore = [[4, 3, 4, 4, 4, 4, 3, 4],
             [4, 4, 4, 4, 4, 4, 4, 4],
             [1, 1, 2, 3, 3, 2, 1, 1],
             [1, 2, 3, 4, 4, 3, 2, 1],
             [1, 2, 3, 4, 4, 3, 2, 1],
             [1, 1, 2, 3, 3, 2, 1, 1],
             [4, 4, 4, 4, 4, 4, 4, 4],
             [4, 3, 4, 4, 4, 4, 3, 4]]

whitePawnScore = [[8, 8, 8, 8, 8, 8, 8, 8],
                  [8, 8, 8, 8, 8, 8, 8, 8],
                  [5, 6, 6, 7, 7, 6, 6, 5],
                  [2, 3, 3, 5, 5, 3, 3, 2],
                  [1, 2, 3, 4, 4, 3, 2, 1],
                  [1, 2, 3, 3, 3, 3, 2, 1],
                  [1, 1, 1, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0]]

blackPawnScore = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 1],
                  [1, 2, 3, 3, 3, 3, 2, 1],
                  [1, 2, 3, 4, 4, 3, 2, 1],
                  [2, 3, 3, 5, 5, 3, 3, 2],
                  [5, 6, 6, 7, 7, 6, 6, 5],
                  [8, 8, 8, 8, 8, 8, 8, 8],
                  [8, 8, 8, 8, 8, 8, 8, 8]]

piecePosScores = {
    'N': knightScore,
    'B': bishopScore,
    'Q': queenScore,
    'R': rookScore,
    "P": whitePawnScore,
    "p": blackPawnScore
}

def get_piece(board: Board, x: int, y: int) -> str:
    """Get piece at position (x,y) with bottom-left coordinate"""
    
    # Valid position
    if (x < 0 or x > 7 or y < 0 or y > 7):
        return ""
    
    pos = f"{ROWS[x]}{COLS[y]}"
    
    chess_piece = board.piece_at(chess.parse_square(pos))
    # Get piece at pos (python-chess format: lower-case for BLACK and upper-case for WHITE)
    if chess_piece:
        piece = chess_piece.symbol()
    else:
        piece = "."
    
    return piece
