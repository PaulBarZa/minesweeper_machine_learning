class Variable:
    def __init__(self, name, domain, nrows, ncols, n_mines):
        self.name = name
        self.domain = list(domain)
        self.current_domain = self.init_current_domain()
        self.value = None
        self.mines_proba = n_mines / (nrows * ncols)

    def init_current_domain(self):
        current_domain = []
        for i in range(len(self.domain)):
            current_domain.append(True)
        return current_domain

    def get_domain(self):
        return list(self.domain)

    def get_current_domain(self):
        values = []
        if self.value != None:
            values.append(self.value)
        else:
            for i, value in enumerate(self.domain):
                if self.current_domain[i]:
                    values.append(value)
        return values

    def in_current_domain(self, value):
        if not value in self.domain:
            return False
        if self.value != None:
            return value == self.value
        else:
            return self.current_domain[self.domain.index(value)]

    def current_domain_size(self):
        if self.value != None:
            return 1
        else:
            return(sum(1 for domain in self.current_domain if domain))

    def remove_domain(self, domain):
        self.current_domain[self.domain.index(domain)] = False

    def get_info(self):
        print("Variable", self.name, self.domain, self.current_domain)


class Constraint:
    def __init__(self, name, variables):
        self.name = name
        self.variables = list(variables)
        self.possible_values = dict()
        self.valid_domain = dict()

    def actualize_variables_value(self):
        values = []
        possible_values = list(self.possible_values.items())

        for i in range(len(possible_values)):
            # If possible value is true
            if possible_values[i][1]:
                # For each possible value
                for val in possible_values[i][0]:
                    values.append(val)

        variables_nb = len(self.variables)
        if len(values) == variables_nb:
            for i in range(variables_nb):
                if self.name == "end_game":
                    self.variables[i].value = 1 - values[i]
                else:
                    self.variables[i].value = values[i]

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
