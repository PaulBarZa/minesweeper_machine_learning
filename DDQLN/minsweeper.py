from math import*
import random
import numpy as np

# Variable to store minesweepers
minsweepersArray = []


class Environment():
    def __init__(self, width, height, n_mines,
                 rewards={'win': 1, 'lose': -1, 'progress': 0.9, 'guess': -0.3, 'no_progress': -0.3}):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.board = self.create_minesweeper()
        self.playing_board = self.generate_player_minesweeper()
        self.rewards = rewards

    # def save_minesweeper(array):
    #     # Ask user if he wants to save his minesweeper
    #     save = input(
    #         "Voulez vous sauvegarder ce démineur (y/n) :")
    #     if (save == "y"):
    #         print("Votre démineur aura l'identifiant %d" %
    #               len(minsweepersArray))
    #         # Add the array to the minesweepers storage array
    #         minsweepersArray.append(array)

    def create_minesweeper(self):
        # Generate an array of n x h (n columns x h lines)
        array = [[0 for col in range(self.ncols)]
                 for row in range(self.nrows)]

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

                    isValid = False

        # Save or note this new minesweeper
        # save_minesweeper(array)
        print(array)
        return array

    def get_player_board(self):
        return self.playing_board

    def generate_player_minesweeper(self):
        # Generate the player minsweeper (n (columns) x h (lines))
        return np.array([[-1 for row in range(self.ncols)] for column in range(self.nrows)])

    # def select_minesweeper():
    #     i = 0
    #     # Display each array (minesweeper) saved
    #     for array in minsweepersArray:
    #         print("-------- Grille identifiant numero % s --------" % i)
    #         display_minesweeper(array)
    #         i = i + 1

    #     # Select the minsweeper
    #     index = int(input("Veuillez rentrer l'identifiant du démineur :"))
    #     return minsweepersArray[index]

    # def verify_continue_game():
    #     # Check if the user want to stop playing or not
    #     isContinue = input("Souhaitez vous recommencer une partie ? (y/n) :")
    #     if isContinue == 'n':
    #         return False
    #     return True

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
                    # If cel is not out of bound, discover it
                    self.playing_board[x][y + 1] = self.board[x][y + 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 1):
                    if (self.board[x][y - 1] == 0 and (self.playing_board[x][y - 1] == -1)):
                        zeroCellsPosition.append([x, y - 1])
                    self.playing_board[x][y - 1] = self.board[x][y - 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y - 1] == 0 and (self.playing_board[x - 1][y - 1] == -1)):
                        zeroCellsPosition.append([x - 1, y - 1])
                    self.playing_board[x - 1][y - 1] = self.board[x - 1][y - 1]

                if (y >= 0 and y <= self.ncols - 2) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y + 1] == 0 and (self.playing_board[x - 1][y + 1] == -1)):
                        zeroCellsPosition.append([x - 1, y + 1])
                    self.playing_board[x - 1][y + 1] = self.board[x - 1][y + 1]

                if (y >= 1 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y - 1] == 0 and (self.playing_board[x + 1][y - 1] == -1)):
                        zeroCellsPosition.append([x + 1, y - 1])
                    self.playing_board[x + 1][y - 1] = self.board[x + 1][y - 1]

                if (y >= 0 and y <= self.ncols - 2) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y + 1] == 0 and (self.playing_board[x + 1][y + 1] == -1)):
                        zeroCellsPosition.append([x + 1, y + 1])
                    self.playing_board[x + 1][y + 1] = self.board[x + 1][y + 1]

                if (y >= 0 and y <= self.ncols - 1) and (x >= 0 and x <= self.nrows - 2):
                    if (self.board[x + 1][y] == 0 and (self.playing_board[x + 1][y] == -1)):
                        zeroCellsPosition.append([x + 1, y])
                    self.playing_board[x + 1][y] = self.board[x + 1][y]

                if (y >= 0 and y <= self.ncols - 1) and (x >= 1 and x <= self.nrows - 1):
                    if (self.board[x - 1][y] == 0 and (self.playing_board[x - 1][y] == -1)):
                        zeroCellsPosition.append([x - 1, y])
                    self.playing_board[x - 1][y] = self.board[x - 1][y]

                # Suppression of the position in the array of zero cells position
                del zeroCellsPosition[zeroCellsPosition.index(position)]

    # def display_minesweeper(map):
    #     # Display a map as a minsweeper
    #     for row in map:
    #         print(" | ".join(str(cell) for cell in row))
    #         print("")

    def is_finished(self, x, y):

        undiscoveredCells = 0
        board = self.playing_board
        board[x][y] = self.board[x][y]
        #  Recover the number of undiscovered cells
        for row in board:
            for cell in row:
                if cell == -1:
                    undiscoveredCells = undiscoveredCells + 1

        if (undiscoveredCells == self.n_mines):
            return True
        return False

    # def init_game():
    #     choice = input("Souhaitez charger un démineur déjà existant ? (y/n) :")

    #     # Check if the user want to load a minsweeper and if there is saved minesweeper
    #     if (choice == "y") and (len(minsweepersArray) > 0):
    #         minsweeper = select_minesweeper()

    #         # Transform the array in matrix
    #         a = np.array(minsweeper)
    #         # Recover the number of rows
    #         rows = a.shape[1]
    #         # Recover the number of columns
    #         columns = a.shape[0]

    #         playerMinsweeper = generate_player_minesweeper(
    #             rows, columns)

    #         return minsweeper, generate_player_minesweeper(rows, columns), rows, columns

    #     else:
    #         print("Aucun démineur n'est sauvegardé")
    #         # Ask user to create his minsweeper
    #         print(
    #             "Indiquer la taille du tableau ainsi que le nombre de bombes")
    #         n = int(input("Longeur :"))
    #         h = int(input("Hauteur :"))
    #         level = input(
    #             "Choisir une difficulté entre simple - intermédiaire - difficile (Entrer : s, i, d): :")
    #         if level.lower() == 's':
    #             # 10% of bombs
    #             b = ceil(0.10 * (n * h))
    #             print("Nombre de bombe(s) : ", b)
    #         elif level.lower() == 'i':
    #             # 15% of bombs
    #             b = ceil(0.15 * (n * h))
    #             print("Nombre de bombe(s) : ", b)
    #         else:
    #             # 30% of bombs
    #             b = ceil(0.30 * (n * h))
    #             print("Nombre de bombe(s) : ", b)
    #         return create_minesweeper(n, h, b), generate_player_minesweeper(n, h), n, h

    # def game():
    #     # Start the game
    #     is_game_running = True

    #     while is_game_running:

    #         minsweeper = create_minesweeper(
    #             self.nrows, self.ncols, self.n_mines)

    #         while True:
    #             # Check if the game is finished
    #             if is_finished(minsweeper, playerMinsweeper) == False:
    #                 print("-------- Grille de jeu --------")
    #                 display_minesweeper(playerMinsweeper)
    #                 # Ask user to select a cell
    #                 print(
    #                     "Indiquer les coordonnées de la case que vous souhaitez découvrir :")
    #                 x = int(input("X (Ligne de 1 à %d):" % n)) - 1
    #                 y = int(input("Y (Colonne de 1 à %d):" % h)) - 1

    #                 # Check if cell is a bomb (if it's a bomb -> game lost)
    #                 if (minsweeper[x][y] == -2):
    #                     print("Perdu")
    #                     print("-------- Votre grille --------")
    #                     display_minesweeper(playerMinsweeper)
    #                     print("-------- Grille de jeu --------")
    #                     display_minesweeper(minsweeper)
    #                     # Ask user if he wants to start new game
    #                     is_game_running = verify_continue_game()
    #                     break

    #                 # Check if cell is already discovered
    #                 elif (playerMinsweeper[x][y] != '-'):
    #                     print("Case déjà découverte, veuillez choisir une autre case")

    #                 # Actualize the player minsweeper
    #                 else:
    #                     playerMinsweeper, minsweeper = discover_cell(
    #                         x, y, playerMinsweeper, minsweeper, n, h)

    #             else:
    #                 print("-------- Votre grille --------")
    #                 display_minesweeper(playerMinsweeper)
    #                 print("Gagné !")
    #                 # Ask user if he wants to start new game
    #                 is_game_running = verify_continue_game()
    #                 break
