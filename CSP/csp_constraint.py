class Constraint:
    def __init__(self, name, variables):
        self.name = name
        self.variables = list(variables)
        self.possible_values = dict()
        self.valid_domain = dict()

    def add_possible_values(self, verify_values):
        # Possible values are values that verify the summ of mines around
        for value in verify_values:
            t = tuple(value)  # ensure we have an immutable tuple
            if not t in self.possible_values:
                self.possible_values[t] = True

            # now put t in as a support for all of the variable values in it
            for i, value in enumerate(t):
                variable = self.variables[i]
                if not (variable, value) in self.valid_domain:
                    self.valid_domain[(variable, value)] = []
                self.valid_domain[(variable, value)].append(t)

    def get_info(self):
        var = []
        for v in self.variables:
            var.append([v.name, v.value, v.domain, v.current_domain])
        print("Constraint", self.name, var, self.possible_values)

    def get_variables(self):
        return list(self.variables)

    def get_possible_values(self):
        return list(self.possible_values)

    def get_sum(self):
        return sum(list(self.possible_values)[0])

    def get_unknown_variables(self):
        unknown_variables = []

        for variable in self.variables:
            if variable.value == None:
                unknown_variables.append(variable)

        return unknown_variables

    def is_valid_domain(self, variable, value):
        if (variable, value) in self.valid_domain:
            for t in self.valid_domain[(variable, value)]:
                if self.tuple_is_valid(t):
                    return True
        return False

    def tuple_is_valid(self, value):
        for i, var in enumerate(self.variables):
            if not var.in_current_domain(value[i]):
                return False
        return True

    def contain_no_cell_variable(self):
        no_cell_variable = False

        for variable in self.get_variables():
            if not variable.is_cell():
                no_cell_variable = True

        return no_cell_variable
