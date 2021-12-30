
import sys
sys.path.insert(1, "../environment")

from environment import Environment
from csp_model import CSP_Model
from csp_solver import CSP_Solver
import random
import numpy as np
from math import ceil

# Prevents groups of cells greater than BOARD_PART_LEN, to avoid RAM issues
BOARD_PART_LEN = 15

# In real game, the first move is free
FREE_FIRST_MOVE = False

GAMES = 10
STATS_EVERY = 10

# Board size
RANDOM_SIZE = True
MIN_SIZE = 5
MAX_SIZE = 25

# Mine density
RANDOM_DENSITY = True
MIN_DENSITY_PERCENT = 5
MAX_DENSITY_PERCENT = 25

# If RANDOM_SIZE is False define :
ROWS = 24
COLS = 24
# If RANDOM_DENSITY is False define :
MINES = 86

if __name__ == "__main__":

    wins_list = []
    index = 0
    for i in range(GAMES):
        index += 1
        # Play on random size boards
        if RANDOM_SIZE:
            ROWS = random.randint(MIN_SIZE, MAX_SIZE)
            COLS = random.randint(MIN_SIZE, MAX_SIZE)
        if RANDOM_DENSITY:
            density = random.randint(MIN_DENSITY_PERCENT, MAX_DENSITY_PERCENT) / 100
            MINES = ceil(density * (ROWS * COLS))

        print("Game : ", i, " Size: ", ROWS, COLS, MINES)

        env = Environment(ROWS, COLS, MINES)
        env.do_first_move(FREE_FIRST_MOVE)
        csp_model = CSP_Model(env, MINES)
        solver = CSP_Solver(env, csp_model, BOARD_PART_LEN)

        is_win = solver.solve()

        if is_win:
            wins_list.append(1)
            print("Win")
        else:
            wins_list.append(0)
            print("Loose")

        if not index % STATS_EVERY:
            win_rate = round(
                np.sum(wins_list[-STATS_EVERY:]) / STATS_EVERY, 2)
            fichier = open("stats.txt", "a")
            fichier.write(
                f"\n - Episode: {index}, Win rate : {win_rate}, Size : {ROWS}x{COLS}x{MINES}, Max length: {BOARD_PART_LEN}")
            fichier.close()
