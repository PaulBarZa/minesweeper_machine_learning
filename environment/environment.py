from math import*
import random
import numpy as np

REWARD_STRUCT = {'win': 1, 'lose': -1, 'progress': 0.9,
                 'guess': -0.4, 'no_progress': -0.3}


class Environment():
    def __init__(self, rows, cols, n_mines):

        self.nrows, self.ncols, self.n_mines = rows, cols, n_mines
        self.ntiles = self.nrows * self.ncols

        self.board = self.create_minesweeper()
        # Player board is a -1 board of nrows x ncols
        self.playing_board = np.ones(
            (self.nrows, self.ncols), dtype=object) * (-1)
        self.coords_array = self.init_coords()

        self.rewards = REWARD_STRUCT

    def init_coords(self):
        coords_array = []

        for i in range(self.nrows):
            for j in range(self.ncols):
                coords_array.append([i, j])

        return coords_array

    def create_minesweeper(self):

        array = np.zeros((self.nrows, self.ncols), dtype=object)

        # Ask user to select bomb(s) position
        for _ in range(self.n_mines):
            valid_coord = True
            while valid_coord:
                # Select a random position for the bomb
                x, y = self.get_random_coords()

                # Check if there is no bomb in this cell
                if not array[x][y] == -2:
                    array[x][y] = -2

                    if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 1):
                        if array[x][y + 1] != -2:
                            array[x][y + 1] += 1  # center right

                    if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 1):
                        if array[x][y - 1] != -2:
                            array[x][y - 1] += 1  # center left

                    if (y >= 1 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                        if array[x - 1][y - 1] != -2:
                            array[x - 1][y - 1] += 1  # top left

                    if (y >= 0 and y <= self.ncols - 2) and (x >= 1 and x <= self.nrows - 1):
                        if array[x - 1][y + 1] != -2:
                            array[x - 1][y + 1] += 1  # top right

                    if (y >= 0 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                        if array[x - 1][y] != -2:
                            array[x - 1][y] += 1  # top center

                    if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 2):
                        if array[x + 1][y + 1] != -2:
                            array[x + 1][y + 1] += 1  # bottom right

                    if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                        if array[x + 1][y - 1] != -2:
                            array[x + 1][y - 1] += 1  # bottom left

                    if (y >= 0 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                        if array[x + 1][y] != -2:
                            array[x + 1][y] += 1  # bottom center

                    valid_coord = False
        return array

    def do_first_move(self, is_random):
        no_valid_cell_found = True
        corner_tried = 0

        # First move is free
        while no_valid_cell_found:
            self.board = self.create_minesweeper()
            self.playing_board = np.ones(
                (self.nrows, self.ncols), dtype=object) * (-1)

            if not is_random and corner_tried < 4:
                row, col = self.get_random_coords()

            else:
                row, col = self.get_random_border_coord(
                    random.randint(0, 3))
                corner_tried += 1

            _, _, no_valid_cell_found, _ = self.discover_cell(row, col)

    def get_cells_around(self, x, y):
        board = self.playing_board
        cells_arround = []

        if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": board[x][y + 1], "x": x, "y": y + 1})

        if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": board[x][y - 1], "x": x, "y": y - 1})

        if (y >= 1 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": board[x - 1][y - 1], "x": x - 1, "y": y - 1})

        if (y >= 0 and y <= self.ncols - 2) and (x >= 1 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": board[x - 1][y + 1], "x": x - 1, "y": y + 1})

        if (y >= 0 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": board[x - 1][y], "x": x - 1, "y": y})

        if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 2):
            cells_arround.append(
                {"value": board[x + 1][y + 1], "x": x + 1, "y": y + 1})

        if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
            cells_arround.append(
                {"value": board[x + 1][y - 1], "x": x + 1, "y": y - 1})

        if (y >= 0 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
            cells_arround.append(
                {"value": board[x + 1][y], "x": x + 1, "y": y})

        return cells_arround

    def discover_cell(self, x, y):
        is_finished = False
        is_win = 0
        cell_value = self.playing_board.tolist()[x][y]

        self.playing_board[x][y] = self.board[x][y]

        if self.playing_board[x][y] == 0:
            self.discover_zero_cells(x, y)

        if self.playing_board[x][y] == -2:  # if lose
            reward = self.rewards['lose']
            is_finished = True

        elif self.is_finished():  # if win
            reward = self.rewards['win']
            is_finished = True
            is_win = 1

        # If select an already known cell
        elif not cell_value == -1.0:
            reward = self.rewards['no_progress']

        else:
            cells_around = self.get_cells_around(x, y)
            # If all cells around are unknown
            if all(v["value"] == -1 for v in cells_around):
                reward = self.rewards['guess']
            else:
                reward = self.rewards['progress']

        return self.playing_board, reward, is_finished, is_win

    def discover_zero_cells(self, X, Y):
        # Store variables in others so they can be changed
        x, y = X, Y
        # Array of position (x, y) where there is only zero
        zero_cells_coords = [[x, y]]

        # While there is not threated postion run this
        while len(zero_cells_coords) > 0:
            for coord in zero_cells_coords:
                x, y = coord[0], coord[1]

                # Check if cells around are not out of bounds
                if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 1):
                    # Check if the cell is a 0 and is unknown for the player
                    if (self.board[x][y + 1] == 0) and (self.playing_board[x][y + 1] == -1):
                        # If yes add it to our zero cells array
                        zero_cells_coords.append([x, y + 1])
                    # If cell is not out of bound, discover it
                    self.playing_board[x][y + 1] = self.board[x][y + 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 1):
                    if (self.board[x][y - 1] == 0) and (self.playing_board[x][y - 1] == -1):
                        zero_cells_coords.append([x, y - 1])
                    self.playing_board[x][y - 1] = self.board[x][y - 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y - 1] == 0) and (self.playing_board[x - 1][y - 1] == -1):
                        zero_cells_coords.append([x - 1, y - 1])
                    self.playing_board[x - 1][y - 1] = self.board[x - 1][y - 1]

                if (y >= 0 and y <= self.ncols - 2) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y + 1] == 0) and (self.playing_board[x - 1][y + 1] == -1):
                        zero_cells_coords.append([x - 1, y + 1])
                    self.playing_board[x - 1][y + 1] = self.board[x - 1][y + 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y - 1] == 0) and (self.playing_board[x + 1][y - 1] == -1):
                        zero_cells_coords.append([x + 1, y - 1])
                    self.playing_board[x + 1][y - 1] = self.board[x + 1][y - 1]

                if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y + 1] == 0) and (self.playing_board[x + 1][y + 1] == -1):
                        zero_cells_coords.append([x + 1, y + 1])
                    self.playing_board[x + 1][y + 1] = self.board[x + 1][y + 1]

                if (y >= 0 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y] == 0) and (self.playing_board[x + 1][y] == -1):
                        zero_cells_coords.append([x + 1, y])
                    self.playing_board[x + 1][y] = self.board[x + 1][y]

                if (y >= 0 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y] == 0) and (self.playing_board[x - 1][y] == -1):
                        zero_cells_coords.append([x - 1, y])
                    self.playing_board[x - 1][y] = self.board[x - 1][y]

                # Suppression of the zero cell because it has been treated
                del zero_cells_coords[zero_cells_coords.index(coord)]

    def is_finished(self):
        if self.remaining_mines(self.playing_board) == self.n_mines:
            return True
        return False

    def remaining_mines(self, board):
        remaining_mines = 0
        for row in board:
            for cell in row:
                if cell == -1:
                    remaining_mines += 1
        # print(remaining_mines)
        # print(board.sum(cell == -1))
        # input("w")
        return remaining_mines

    def get_random_border_coord(self, choice):
        starting_coords = [[0, 0], [0, self.ncols - 1],
                           [self.nrows - 1, 0], [self.nrows - 1, self.ncols - 1]]
        return starting_coords[choice][0], starting_coords[choice][1]

    def get_random_coords(self):
        return random.randint(0, self.nrows - 1), random.randint(0, self.ncols - 1)

    def get_board(self):
        return self.board

    def get_player_board(self):
        return self.playing_board
