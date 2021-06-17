import sys
sys.path.insert(1, "environment")

from csp_model import *
import itertools
import numpy as np


class CSP_Solver():
    def __init__(self, env, model, max_part_length):
        self.env = env
        self.model = model
        self.max_part_length = max_part_length

    def solve(self):
        done = False

        while not done:

            find_cell, cells = self.get_cell()
            board = self.env.get_player_board()

            if not find_cell:

                cell = self.get_best_cell()
                board, _, done, _ = self.env.discover_cell(
                    cell[0], cell[1])

            else:
                for cell in cells:
                    board, _, done, _ = self.env.discover_cell(
                        cell[0], cell[1])

            if self.env.remaining_mines(board) == self.model.nmines:
                done = True

        return self.env.remaining_mines(board) == self.model.nmines

    def get_cell(self):

        find_cell = False
        cells = []
        board = self.env.get_player_board()

        self.model.actualize_model()

        self.gac_propagotor()

        for var in self.model.get_variables():
            remaining_mines = self.model.remaining_mines()
            var.actualize_variable(remaining_mines)

        self.set_propagator()

        for var in self.model.get_variables():
            remaining_mines = self.model.remaining_mines()
            var.actualize_variable(remaining_mines)

        for variable in self.model.get_variables():

            try:
                cell = variable.name.split()
                row = int(cell[0])
                col = int(cell[1])
            except:
                # continue if it's not a board variable or end_game constraint.
                continue

            if variable.value == 0:
                if board[row][col] == -1:
                    find_cell = True
                    cells.append([row, col])

        return find_cell, cells

    def get_best_cell(self):

        self.calcul_cells_proba()
        variables = self.model.get_variables()

        best_var = variables[0]

        for var in variables:
            if var.value == None and var.is_cell():
                if var.mines_proba < best_var.mines_proba:
                    best_var = var
        cell = best_var.name.split()

        return [int(cell[0]), int(cell[1])]

    def gac_propagotor(self):

        constraints_copy = self.model.get_constraints().copy()

        index = 0
        while index < len(constraints_copy):

            constraint = constraints_copy[index]
            variables = constraint.get_variables()

            for variable in variables:

                current_domain = variable.get_current_domain()

                found = False
                for domain in current_domain:

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
                        self.model.constraints_with_variables[variable])
                    for constraint in constraints_:
                        if constraint not in constraints_copy[index:]:
                            constraints_copy.append(constraint)
            index += 1

            if index > 1500:  # TODO find why sometimes infinite loop
                return True

        return True

    def set_propagator(self):

        constraints_copy = self.model.get_constraints().copy()

        for constraint in constraints_copy:
            if constraint.name == "":
                unknown_variables = constraint.get_unknown_variables()
                for unknown_var in unknown_variables:
                    verify_variables_domain(unknown_var)
    #
    # ---------------- Probalities --------------------------
    #

    def calcul_cells_proba(self):

        groups = self.get_groups()

        for group in groups:
            equalities = []
            registerd_eq = []

            uniq_var = get_uniq_var(group)

            domains = []
            for var in uniq_var:
                domains.append(var.domain)

            for constraint in group:
                if constraint.contain_no_cell_variable():
                    continue

                c_variables = constraint.get_variables()
                possible_values = constraint.get_possible_values()

                if len(c_variables) > 0 and len(possible_values) > 0:

                    equality = [[*c_variables], sum(possible_values[0])]

                    if not equality in registerd_eq:
                        equalities.append(equality)
                        registerd_eq.append(equality)

            results = list(itertools.product(*domains))

            # Filter admitted results
            admitted_results = []
            for equality in equalities:
                index_var = []
                for var in equality[0]:
                    index_var.append(uniq_var.index(var))

                for res in results:
                    res_values = 0
                    for index in index_var:
                        res_values += res[index]

                    if res_values == equality[1]:
                        admitted_results.append(res)

            # Calcul the proba if there is admittes results
            if len(admitted_results) > 0:
                proba_by_var = [round(sum(res[i] for res in admitted_results) / len(admitted_results), 2)
                                for i in range(len(admitted_results[0]))]

                for i in range(len(uniq_var)):
                    uniq_var[i].mines_proba = proba_by_var[i]

    def get_groups(self):
        groups = []
        registered_variables = []

        for variable in self.model.get_variables():
            if variable in registered_variables:
                continue
            # Here we want unknown variables who already have a constraint
            starting_group = self.model.constraints_with_variables[variable]
            if starting_group and variable.value == None:

                group = self.find_constraint_group(starting_group)

                group_list = self.split_group(group)

                for g in group_list:
                    for constraint in g:
                        for var in constraint.get_variables():
                            if not var in registered_variables:
                                registered_variables.append(var)

                    groups.append(g)

        return groups

    def find_constraint_group(self, group):
        group_variables = []
        find_constraint = False

        # Recup all variables of the current group
        for constraint in group:
            variables = constraint.get_variables()
            # Recup the uncommon variables
            common_vars = set(variables) & set(group_variables)
            uncommon_vars = set(variables) - common_vars
            # Add the uncommon variables in the group variables
            for uncommon_var in uncommon_vars:
                # Limit the number of unique variables in a group
                if uncommon_var.value == None:
                    group_variables.append(uncommon_var)

        # Add new constraint corresponding to each variables in the group
        for group_var in group_variables:

            constraints = self.model.constraints_with_variables[group_var]

            # If there is at least one constraint corresponding to the variable append it to the group
            if len(constraints) > 0:
                for c in constraints:
                    if not c in group:
                        if c.name != "":
                            group.append(c)
                            find_constraint = True

        if find_constraint:
            self.find_constraint_group(group)

        return group

    def split_group(self, group):
        # Split the goup if he is too big (too many unique variables)
        groups = [group]
        uniq_vars_len = len(get_uniq_var(group))
        split_in = 1
        should_split = False
        is_only_endgame = False

        if uniq_vars_len > self.max_part_length:
            should_split = True

        while should_split and not is_only_endgame:
            split_in += 1
            # Split in X groups
            splitted_group = list(np.array_split(np.array(group), split_in))
            groups = []
            should_split = False
            is_only_endgame = False

            for s_group in splitted_group:
                for c in s_group:
                    if c.name != "end_game":
                        is_only_endgame = True

                if len(get_uniq_var(s_group)) > self.max_part_length:
                    should_split = True
                groups.append(s_group)

        return groups


def get_uniq_var(group):
    uniq_var = []
    for c in group:
        if c.contain_no_cell_variable():
            continue
        for v in c.get_variables():
            if not v in uniq_var:
                uniq_var.append(v)

    return uniq_var


def verify_variables_domain(unknown_variable):
    if unknown_variable.current_domain_size() == 1:
        if unknown_variable.value == None:
            unknown_variable.value = unknown_variable.get_current_domain()[
                0]
