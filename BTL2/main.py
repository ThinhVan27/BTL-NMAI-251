# Run and Evaluate game 
import chess
import time
import random

if __name__ == "__main__":
    board = chess.Board()
    print("Init game", board)
    t = 20
    while (t > 0):
        move = random.choice(list(board.legal_moves))
        board.push(move)
        print("[INFO] Game state")
        print(board)
        t -= 1