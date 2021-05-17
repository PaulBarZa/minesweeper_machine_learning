
import sys
sys.path.insert(1, "environment")

from environment import Environment
from cspinit import CSP_Model
import random
import numpy as np


def solve(env, csp_model, mines):
    done = False
    board = env.get_player_board()

    while not done:

        find_cell, done = get_cell(env, csp_model)

        if not find_cell:
            show_or_flag = True
            while show_or_flag:
                row = random.randint(0, env.nrows - 1)
                col = random.randint(0, env.ncols - 1)
                if board[row][col] == -1:
                    show_or_flag = False
            _, _, done, _ = env.discover_cell(row, col, False)
        # print("Find cell ", find_cell)
    return env.remaining_mines(board) == mines


def get_cell(env, csp_model):

    find_cell, done = False, False
    board = env.get_player_board()

    gac_propagotor(csp_model)

    for variable in csp_model.get_variables():

        try:
            cell = variable.name.split()
            row = int(cell[0])
            col = int(cell[1])
        except:
            # continue if it's not a board variable or end_game constraint.
            continue

        if variable.value == 0:
            if board[row][col] == -1:
                _, _, done, _ = env.discover_cell(row, col, False)
                find_cell = True

    return find_cell, done


def gac_propagotor(csp_model):

    csp_model.actualize_model()
    constraints_copy = csp_model.get_constraints().copy()

    index = 0
    while index < len(constraints_copy):

        constraint = constraints_copy[index]
        variables = constraint.get_variables()

        for variable in variables:

            current_domain = variable.get_current_domain()

            found = False
            for domain in current_domain:

                constraint.actualize_variables_value()

                if constraint.is_valid_domain(variable, domain):
                    continue
                else:
                    found = True
                    # Remove domain from current domain
                    variable.remove_domain(domain)
                    if not variable.current_domain_size():
                        constraints_copy = []
                        return False
            # Add constraint who have the removed var to loop on it again
            if found:
                constraints_ = list(
                    csp_model.constraints_with_variables[variable])
                for constraint in constraints_:
                    if constraint not in constraints_copy[index:]:
                        constraints_copy.append(constraint)
        index += 1

        if index > 1500:
            return True

    return True


EPISODE = 1000
STATS_EVERY = EPISODE
ROWS = 16
COLS = 16
MINES = 25

if __name__ == "__main__":
    wins_list = []
    index = 0
    for i in range(EPISODE):
        index += 1

        done = True
        while done:
            env = Environment(ROWS, COLS, MINES)
            x = random.randint(0, env.nrows - 1)
            y = random.randint(0, env.ncols - 1)
            _, _, done, _ = env.discover_cell(x, y, False)

        csp_model = CSP_Model(env)

        is_win = solve(env, csp_model, MINES)

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
