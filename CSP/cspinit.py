import sys
sys.path.insert(1, "environment")

from cspsolver import Variable, Constraint
import itertools


class CSP_Model():
    def __init__(self, env):
        self.env = env
        self.board = []
        self.constraints_with_variables = dict()
        self.actualize_model()

    def actualize_model(self):

        self.board = self.env.get_player_board()

        variables = self.actualize_variables()

        constraints, unknown_cells = self.actualize_constraints(variables)

        # Optimization
        if len(unknown_cells) <= 20:
            # print(self.env.remaining_mines(self.board))
            constraints.append(
                ["end_game", unknown_cells, self.env.remaining_mines(self.board)])

        # Sort constraints from biggest to smallest list of variables
        constraints.sort(key=lambda constraint: len(constraint[1]))

        constraints = self.apply_simple_propagators(constraints)

        # Sort constraints from biggest to smallest list of variables
        constraints.sort(key=lambda x: len(x[1]))

        constraints = self.apply_set_propagators(constraints)

        for con in constraints:
            constraint = Constraint(con[0], con[1])
            possible_values = self.possible_values(con[1], con[2])
            constraint.add_possible_values(possible_values)
            self.add_constraint(constraint)

    def actualize_variables(self):
        self.variables = []

        variables = []
        # Initialize all variables in board
        for row in range(self.env.nrows):
            row_variables = []
            for col in range(self.env.ncols):
                name = str(row) + " " + str(col)
                if self.is_cell_show(row, col):
                    possible_val = [0]
                elif self.is_cell_flagged(row, col):
                    possible_val = [1]
                else:
                    possible_val = [0, 1]
                var = Variable(name, possible_val, self.env.nrows,
                               self.env.ncols, self.env.n_mines)
                row_variables.append(var)
                self.add_variable(var)
            variables.append(row_variables)

        return variables

    def actualize_constraints(self, variables):
        self.constraints = []

        constraints = []
        unknown_cells = []
        for row in range(self.env.nrows):
            for col in range(self.env.ncols):
                if self.is_cell_show(row, col):
                    variables[row][col].value = 0
                else:
                    unknown_cells.append(variables[row][col])

                # If cell should be shown
                if self.is_cell_show(row, col) and self.board[row][col] != 0:
                    cells_around = self.env.get_cells_around(
                        row, col)
                    variables_around = []
                    # Multiply by 8 because board unit is 0.125
                    cell_sum = self.board[row][col] * 8

                    for cell in cells_around:
                        # If cell is flagged
                        if cell['value'] == -16:
                            cell_sum -= 1
                        # If cell is unknown
                        if cell['value'] == -8:
                            variables_around.append(
                                variables[cell['x']][cell['y']])
                    name = str(row) + " " + str(col)

                    if variables_around:
                        constraints.append([name, variables_around, cell_sum])

        return constraints, unknown_cells

    def apply_simple_propagators(self, constraints):
        # Reduce constraint's scope.
        # ex: c1=[v1,v2], c2=[v1,v2,v3] => reduce c2 to [v3]
        for i in range(len(constraints) - 1):
            con1 = constraints[i]
            for j in range(i + 1, len(constraints)):
                con2 = constraints[j]
                if set(con1[1]) == set(con2[1]):
                    continue
                # If all elements of cons1 are in cons2
                if set(con1[1]) & set(con2[1]) == set(con1[1]):
                    # Reduce cons2 as cons2 minus common elements btw both
                    con2[1] = list(set(con2[1]).difference(set(con1[1])))
                    # Reduce the sum
                    con2[2] = con2[2] - con1[2]

        return constraints

    def apply_set_propagators(self, constraints):
        new_constraints = []
        already_in_common = []
        common_var_list = []

        # Add new constraints if two constraints has at least two same variables in scope.
        # Create a new variable for overlap variables.
        # ex: c1=[v1,v2,v3], c2=[v2,v3,v4] => add c3=[v1,v2v3], c4=[v4, v2v3]. v2v3 is a new variable.
        for i in range(len(constraints) - 1):
            con1 = constraints[i]
            for j in range(i + 1, len(constraints)):
                con2 = constraints[j]
                if set(con1[1]) == set(con2[1]):
                    continue
                if 1 < len(set(con1[1]) & set(con2[1])):
                    # Recup the common variables
                    common_vars = set(con1[1]) & set(con2[1])
                    con1_uncommon_var = set(con1[1]) - common_vars
                    con2_uncommon_var = set(con2[1]) - common_vars
                    name = ""

                    if not common_vars in already_in_common:
                        for variable in common_vars:
                            name += variable.name + ", "
                        name = "(" + name + ")"
                        var = Variable(name, list(
                            range(len(common_vars) + 1)), self.env.nrows, self.env.ncols, self.env.n_mines)
                        self.add_variable(var)
                        common_var_list.append(var)
                        already_in_common.append(common_vars)
                    else:
                        index = already_in_common.index(common_vars)
                        var = common_var_list[index]

                    con1_uncommon_var.add(var)
                    con2_uncommon_var.add(var)
                    new_constraints.append(
                        ["", list(con1_uncommon_var), con1[2]])
                    new_constraints.append(
                        ["", list(con2_uncommon_var), con2[2]])

        constraints.extend(new_constraints)
        return constraints

    def add_constraint(self, constraint):
        for variable in constraint.variables:
            self.constraints_with_variables[variable].append(constraint)
        self.constraints.append(constraint)

    def add_variable(self, variable):
        self.variables.append(variable)
        self.constraints_with_variables[variable] = []

    def possible_values(self, variables, sum1):

        domains = []
        for variable in variables:
            domains.append(variable.get_domain())
        domains_list = list(itertools.product(*domains))
        possible_values = []
        for domain in domains_list:
            if sum(domain) == sum1:
                possible_values.append(domain)

        return possible_values

    # def get_csp_info(self):
    #     print(self.board)
    #     for i in range(3):
    #         print(self.variables[i].get_info())
    #         print(self.constraints[i].get_info())

    def get_variables(self):
        return list(self.variables)

    def get_constraints(self):
        return self.constraints

    def is_cell_show(self, x, y):
        if self.board[x][y] != -1 and self.board[x][y] != -2:
            return True
        return False

    def is_cell_flagged(self, x, y):
        if self.board[x][y] == -2:
            return True
        return False
