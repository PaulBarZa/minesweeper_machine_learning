
import sys
sys.path.insert(1, "environment")

from environment import Environment
from csp_model import CSP_Model
from csp_solver import CSP_Solver
import random
import numpy as np
from math import ceil

# Prevents groups of cells greater than X when calculating probabilities
BOARD_PART_LEN = 15

EPISODE = 250
STATS_EVERY = 250

RANDOM_SIZE = False
MIN_SIZE = 5
MAX_SIZE = 25

RANDOM_DENSITY = False
MIN_DENSITY = 5  # In percent
MAX_DENSITY = 25

ROWS = 24
COLS = 24
MINES = 86

if __name__ == "__main__":

    wins_list = []
    index = 0
    for i in range(EPISODE):
        index += 1
        # Play on random size boards
        if RANDOM_SIZE:
            ROWS = random.randint(MIN_SIZE, MAX_SIZE)
            COLS = random.randint(MIN_SIZE, MAX_SIZE)
        if RANDOM_DENSITY:
            density = random.randint(MIN_DENSITY, MAX_DENSITY) / 100
            MINES = ceil(density * (ROWS * COLS))

        print("Game : ", i, " Size: ", ROWS, COLS, MINES)

        env = Environment(ROWS, COLS, MINES)
        env.do_first_move(False)
        csp_model = CSP_Model(env, MINES)
        solver = CSP_Solver(env, csp_model, BOARD_PART_LEN)

        is_win = solver.solve()

        if is_win:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if not index % STATS_EVERY:
            win_rate = round(
                np.sum(wins_list[-STATS_EVERY:]) / STATS_EVERY, 2)
            fichier = open("CSP/stats.txt", "a")
            fichier.write(
                f"\n - Episode: {index}, Win rate : {win_rate}, Size : {ROWS}x{COLS}x{MINES}, Max length: {BOARD_PART_LEN}")
            fichier.close()
