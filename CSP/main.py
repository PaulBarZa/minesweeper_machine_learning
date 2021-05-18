
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
        print(board)
        print("---------")
        find_cell, done = get_cell(env, csp_model)

        if not find_cell:

            row, col = get_best_cell_proba(csp_model)
            _, _, done, _ = env.discover_cell(row, col, False)

        # print("Find cell ", find_cell)
    return env.remaining_mines(board) == mines


def get_cell(env, csp_model):

    find_cell, done = False, False
    board = env.get_player_board()

    csp_model.actualize_model()

    gac_propagotor(csp_model)

    set_propagator(csp_model)

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


def set_propagator(csp_model):

    constraints_copy = csp_model.get_constraints().copy()

    for constraint in constraints_copy:
        if constraint.name == "":
            # print("Constraint name's " " ")
            # constraint.get_info()
            unknown_variables = constraint.get_unknown_variables()
            for unknown_var in unknown_variables:
                verify_variables_domain(unknown_var)


def verify_variables_domain(unknown_variable):
    if unknown_variable.current_domain_size() == 1:
        if unknown_variable.value == None:
            unknown_variable.value = unknown_variable.get_current_domain()[0]


def get_best_cell_proba(csp_model):

    constraints_copy = csp_model.get_constraints().copy()
    variables = actualize_variables_proba(constraints_copy)

    best_variable = variables[0]

    for variable in variables:
        if variable.mines_proba < best_variable.mines_proba:
            best_variable = variable

    cell = best_variable.name.split()

    return cell[0], cell[1]


def actualize_variables_proba(constraints):
    index = 0
    while index < len(constraints):

        constraint = constraints[index]

        possible_values = list(constraint.possible_values)
        if not len(possible_values) or constraint.name == "":
            continue
        mines_number = sum(possible_values[0])
        variable_number = len(constraint.variables)
        proba = round(mines_number / variable_number, 2)

        variables = constraint.get_variables()

        modify = False
        for variable in variables:
            if variable.mines_proba != proba:
                variable.mines_proba = proba
                modify = True

        if modify:
            constraints_ = list(
                csp_model.constraints_with_variables[variable])
            for constraint in constraints_:
                if constraint not in constraints[index:]:
                    constraints.append(constraint)

        index += 1


def get_starting_coord(choice, nrows, ncols):
    starting_coords = [[0, 0], [0, ncols - 1],
                       [nrows - 1, 0], [nrows - 1, ncols - 1]]
    return starting_coords[choice][0], starting_coords[choice][1]


EPISODE = 1000
STATS_EVERY = EPISODE
ROWS = 10
COLS = 10
MINES = 10

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

            _, _, done, _ = env.discover_cell(row, col, False)

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
