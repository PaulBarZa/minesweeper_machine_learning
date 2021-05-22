
import sys
sys.path.insert(1, "environment")

from environment import Environment
from csp_model import CSP_Model
import random
import numpy as np


def get_starting_coord(choice, nrows, ncols):
    starting_coords = [[0, 0], [0, ncols - 1],
                       [nrows - 1, 0], [nrows - 1, ncols - 1]]
    return starting_coords[choice][0], starting_coords[choice][1]


EPISODE = 1_000
STATS_EVERY = EPISODE
ROWS = 6
COLS = 6
MINES = 6

if __name__ == "__main__":
    wins_list = []
    index = 0
    for i in range(EPISODE):
        print("Game : ", i)
        index += 1

        env = Environment(ROWS, COLS, MINES)

        done = True
        corner_mined = 0
        while done:
            if corner_mined > 3:
                row = random.randint(0, env.nrows - 1)
                col = random.randint(0, env.ncols - 1)

            else:
                starting_choice = random.randint(0, 3)
                row, col = get_starting_coord(
                    starting_choice, env.nrows, env.ncols)

                corner_mined += 1

            _, _, done, _ = env.discover_cell(row, col)

        csp_model = CSP_Model(env, MINES)

        is_win = csp_model.solve()

        if is_win:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if not index % STATS_EVERY:
            win_rate = round(
                np.sum(wins_list[-STATS_EVERY:]) / STATS_EVERY, 2)
            fichier = open("CSP/stats.txt", "a")
            fichier.write(
                f"\n - Episode: {index}, Win rate : {win_rate}, Size : {ROWS}x{COLS}x{MINES}")
            fichier.close()
