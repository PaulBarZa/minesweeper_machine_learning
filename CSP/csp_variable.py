class Variable:
    def __init__(self, name, domain, nrows, ncols, n_mines):
        self.name = name
        self.domain = list(domain)
        self.current_domain = self.init_current_domain()
        self.value = None
        self.nrows, self.ncols = nrows, ncols
        self.mines_proba = n_mines / (nrows * ncols)

    def init_current_domain(self):
        current_domain = []
        for i in range(len(self.domain)):
            current_domain.append(True)
        return current_domain

    def actualize_variable(self, remaining_mines):
        self.actualize_variable_value()
        self.actualize_variable_proba(remaining_mines)

    def actualize_variable_value(self):
        true_value_index = []

        current_domain = self.current_domain

        # Recup the index of admitted current values
        for i in range(len(current_domain)):
            if current_domain[i]:
                true_value_index.append(i)

        # If only one admitted current value actualize variable value
        if self.value == None and len(true_value_index) == 1:
            self.value = self.domain[true_value_index[0]]
        # # If there is no admitted current value actualize variable value to 0
        # if self.value == None and len(true_value_index) == 1:
        #     self.value = 0

    def actualize_variable_proba(self, remaining_mines):
        if self.value == None:
            self.mines_proba = remaining_mines / (self.nrows * self.ncols)
        # Here proba for flagged or discovered is one because we don't want to consider them
        else:
            self.mines_proba = 1

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
        try:
            self.current_domain[self.domain.index(domain)] = False
        except:
            print("Not in the domain list")

    def is_cell(self):
        try:
            cell = self.name.split()
            _, _ = int(cell[0]), int(cell[1])
        except:
            return False

        return True

    def get_info(self):
        print("Variable", self.name, self.domain,
              self.current_domain, self.mines_proba)
