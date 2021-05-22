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
        try:
            self.current_domain[self.domain.index(domain)] = False
        except:
            print("Not in the domain list")

    def get_info(self):
        print("Variable", self.name, self.domain, self.current_domain)
