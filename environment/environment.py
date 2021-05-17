from math import*
import random
import numpy as np


class Environment():
    def __init__(self, width, height, n_mines,
                 rewards={'win': 1, 'lose': -1, 'progress': 0.9, 'guess': -0.3, 'no_progress': -0.9}):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.board = self.create_minesweeper()
        self.playing_board = self.generate_player_minesweeper()
        self.rewards = rewards

    def create_minesweeper(self):
        # Generate an array of n x h (n columns x h lines)
        grid = [[0 for col in range(self.ncols)]
                for row in range(self.nrows)]
        array = np.array(grid, dtype=float)
        # Ask user to select bomb(s) position
        for num in range(self.n_mines):
            isValid = True
            while isValid:
                # Select a random position for the bomb
                x = random.randint(0, self.nrows - 1)
                y = random.randint(0, self.ncols - 1)

                # Check if there is no bomb in this cell
                if array[x][y] != -2:
                    array[x][y] = -2

                    if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 1):
                        if array[x][y + 1] != -2:
                            array[x][y + 1] += 0.125  # center right

                    if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 1):
                        if array[x][y - 1] != -2:
                            array[x][y - 1] += 0.125  # center left

                    if (y >= 1 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                        if array[x - 1][y - 1] != -2:
                            array[x - 1][y - 1] += 0.125  # top left

                    if (y >= 0 and y <= self.ncols - 2) and (x >= 1 and x <= self.nrows - 1):
                        if array[x - 1][y + 1] != -2:
                            array[x - 1][y + 1] += 0.125  # top right

                    if (y >= 0 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                        if array[x - 1][y] != -2:
                            array[x - 1][y] += 0.125  # top center

                    if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 2):
                        if array[x + 1][y + 1] != -2:
                            array[x + 1][y + 1] += 0.125  # bottom right

                    if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                        if array[x + 1][y - 1] != -2:
                            array[x + 1][y - 1] += 0.125  # bottom left

                    if (y >= 0 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                        if array[x + 1][y] != -2:
                            array[x + 1][y] += 0.125  # bottom center

                    isValid = False
        return array

    def get_board(self):
        return self.board

    def get_player_board(self):
        return self.playing_board

    def generate_player_minesweeper(self):
        # Generate the player minsweeper (n (columns) x h (lines))
        return np.array([[-1 for row in range(self.ncols)] for column in range(self.nrows)], dtype=float)

    def get_cells_around(self, x, y):
        array = self.playing_board
        cells_arround = []

        if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": array[x][y + 1] * 8, "x": x, "y": y + 1})

        if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": array[x][y - 1] * 8, "x": x, "y": y - 1})

        if (y >= 1 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": array[x - 1][y - 1] * 8, "x": x - 1, "y": y - 1})

        if (y >= 0 and y <= self.ncols - 2) and (x >= 1 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": array[x - 1][y + 1] * 8, "x": x - 1, "y": y + 1})

        if (y >= 0 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
            cells_arround.append(
                {"value": array[x - 1][y] * 8, "x": x - 1, "y": y})

        if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 2):
            cells_arround.append(
                {"value": array[x + 1][y + 1] * 8, "x": x + 1, "y": y + 1})

        if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
            cells_arround.append(
                {"value": array[x + 1][y - 1] * 8, "x": x + 1, "y": y - 1})

        if (y >= 0 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
            cells_arround.append(
                {"value": array[x + 1][y] * 8, "x": x + 1, "y": y})

        return cells_arround

    def discover_cell(self, x, y, guess_move):
        done = False
        is_win = 0

        if guess_move:
            # guess move should always be negative reward
            reward = self.rewards['guess']
            if self.board[x][y] == -2:  # if lose
                done = True
            elif self.is_finished(x, y):  # if win
                done = True
                is_win = 1

        elif self.board[x][y] == -2:  # if lose
            reward = self.rewards['lose']
            done = True

        elif self.is_finished(x, y):  # if win
            reward = self.rewards['win']
            done = True
            is_win = 1

        else:  # if select a correct case
            if self.playing_board[x][y] == -1:
                reward = self.rewards['progress']
            else:
                reward = self.rewards['no_progress']

        self.playing_board[x][y] = self.board[x][y]

        if self.playing_board[x][y] == 0:
            self.discover_zero_cells(x, y)

        return self.playing_board, reward, done, is_win

    def discover_zero_cells(self, X, Y):
        # Store variables in others so they can be changed
        x = X
        y = Y
        # Array of position (x, y) where there is only zero
        zeroCellsPosition = [[x, y]]

        # While there is not threated postion run this
        while len(zeroCellsPosition) > 0:
            for position in zeroCellsPosition:
                # Get the new postion values
                x = position[0]
                y = position[1]

                # Check if cells around are not out of bounds
                if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 1):
                    # Check if the cell is a 0 and a uncovered cell
                    if (self.board[x][y + 1] == 0) and (self.playing_board[x][y + 1] == -1):
                        # If yes add it to our zero position cells array
                        zeroCellsPosition.append([x, y + 1])
                    # If cell is not out of bound, discover it
                    self.playing_board[x][y + 1] = self.board[x][y + 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 1):
                    if (self.board[x][y - 1] == 0) and (self.playing_board[x][y - 1] == -1):
                        zeroCellsPosition.append([x, y - 1])
                    self.playing_board[x][y - 1] = self.board[x][y - 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y - 1] == 0) and (self.playing_board[x - 1][y - 1] == -1):
                        zeroCellsPosition.append([x - 1, y - 1])
                    self.playing_board[x - 1][y - 1] = self.board[x - 1][y - 1]

                if (y >= 0 and y <= self.ncols - 2) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y + 1] == 0) and (self.playing_board[x - 1][y + 1] == -1):
                        zeroCellsPosition.append([x - 1, y + 1])
                    self.playing_board[x - 1][y + 1] = self.board[x - 1][y + 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y - 1] == 0) and (self.playing_board[x + 1][y - 1] == -1):
                        zeroCellsPosition.append([x + 1, y - 1])
                    self.playing_board[x + 1][y - 1] = self.board[x + 1][y - 1]

                if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y + 1] == 0) and (self.playing_board[x + 1][y + 1] == -1):
                        zeroCellsPosition.append([x + 1, y + 1])
                    self.playing_board[x + 1][y + 1] = self.board[x + 1][y + 1]

                if (y >= 0 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y] == 0) and (self.playing_board[x + 1][y] == -1):
                        zeroCellsPosition.append([x + 1, y])
                    self.playing_board[x + 1][y] = self.board[x + 1][y]

                if (y >= 0 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y] == 0) and (self.playing_board[x - 1][y] == -1):
                        zeroCellsPosition.append([x - 1, y])
                    self.playing_board[x - 1][y] = self.board[x - 1][y]

                # Suppression of the position in the array of zero cells position
                del zeroCellsPosition[zeroCellsPosition.index(position)]

    def is_finished(self, x, y):
        board = self.playing_board
        board[x][y] = self.board[x][y]

        if self.remaining_mines(board) == self.n_mines:
            return True
        return False

    def remaining_mines(self, board):
        remaining_mines = 0
        for row in board:
            for cell in row:
                if cell == -1:
                    remaining_mines += 1
        return remaining_mines
