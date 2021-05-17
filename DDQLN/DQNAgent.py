import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os.path
from tensorflow import keras
from random import randint
from random import sample
from collections import deque
from main import*


# Exploration settings
epsilon = 0.01
# epsilon = 1
EPSILON_DECAY = 0.9995
# EPSILON_DECAY = 1
EPSILON_MIN = 0.01

# Deep Q-learning model parameters
learn_rate = 0.0001
# learn_rate = 0
LEARN_DECAY = 0.99995
# LEARN_DECAY = 1
LEARN_MIN = 0.0001
DISCOUNT = 0.0

# Memory & Target
SWAP_TARGET = 5
BATCH_SIZE = 32
REPLAY_MIN_SIZE = 500
REPLAY_MAX_SIZE = 50_000

# Board size to train on
WORKING_SIZE = 3
NB_ACTIONS = WORKING_SIZE * WORKING_SIZE


class DQNAgent(object):
    def __init__(self, env):
        # print("Num GPUs Available: ", len(
        #     tf.config.list_physical_devices('GPU')))
        # Environment
        self.env = env
        self.action_list = self.init_action_list()
        # Deep Q-learning Parameters
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        # Init model and target model
        self.model = self.load_model()
        self.target_model = self.load_target_model()
        # Model weights are randomely defined, and we don't want that
        # self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = self.load_replay_memory()
        self.target_update_counter = 0

    def set_env(self, env):
        self.env = env

    #
    # ----- Action & Prediction -----
    #
    def init_action_list(self):
        # #
        # Init an action list corresponding to the different coords of a board part
        # #
        action_list = []

        for i in range(WORKING_SIZE):
            for j in range(WORKING_SIZE):
                action_list.append([i, j])

        return action_list

    def get_prediction(self):
        # #
        # Get a prediction
        # #
        guess_move = False
        # Random selection b/w exploiration or exploitation
        rand = np.random.random()
        # Explore
        if rand < self.epsilon:
            prediction = self.get_random_prediction()

       # Exploit
        else:
            cells_predictions = self.get_cells_predictions()
            prediction = self.select_best_prediction(cells_predictions)

        if prediction["move"] == NB_ACTIONS:
            guess_move = True

        return prediction, guess_move

    def select_best_prediction(self, predictions):
        # #
        # Select best prediction in a list of predictions
        # #
        selected_prediction = predictions[0][0]

        for row in predictions:
            for prediction in row:
                if prediction["value"] > selected_prediction["value"]:
                    selected_prediction = prediction

        return selected_prediction

    def get_cells_predictions(self):
        # #
        # Get the best predictions for each cell
        # #
        parts_predictions = self.get_board_parts_prediction()

        # Create array of same size of our board, and add a default value of prediction in each case
        cells_predictions = []
        empty_prediction = {
            "value": -100,
        }
        for i in range(self.env.nrows):
            row = []
            for j in range(self.env.ncols):
                row.append(empty_prediction)
            cells_predictions.append(row)

        # For each cell of our array
        for i in range(self.env.nrows):
            for j in range(self.env.ncols):
                prediction_list_values = []

                # Check all predictions for this cell and select the predictions with coord value [i, j]
                for prediction in parts_predictions:
                    if prediction['coords'] == [i, j]:
                        if cells_predictions[i][j]["value"] == -100:
                            cells_predictions[i][j] = prediction
                        if prediction['value'] > cells_predictions[i][j]["value"]:
                            # Select the best prediction for the [i, j] cell
                            cells_predictions[i][j] = prediction
                        # Get all predictions for the [i, j] cell
                        prediction_list_values.append(prediction["value"])

                # Replace value of the best prediction for [i, j] by the average value of all the predictions
                cells_predictions[i][j]["value"] = sum(
                    prediction_list_values) / len(prediction_list_values)
        # print(cells_predictions)
        return cells_predictions

    def get_board_parts_prediction(self):
        # #
        # Get predictions for each cells in each part of the board (WORKING_SIZE x WORKING_SIZE parts)
        # #
        parts_predictions = []
        # For each WORKING_SIZE x WORKING_SIZE parts of the board
        for col_index in range(self.env.ncols - (WORKING_SIZE - 1)):
            for row_index in range(self.env.nrows - (WORKING_SIZE - 1)):

                state = Get_board_part(self.env.get_player_board(
                ), WORKING_SIZE, row_index, col_index).reshape(-1, NB_ACTIONS)

                moves = self.model_predict(state)

                move_action = 0
                for move in moves[0]:
                    prediction = {
                        "state": state,
                        "move": move_action,
                        "value": move.numpy(),
                        "part_coords": [row_index, col_index],
                    }
                    # If prediction isn't a predict move, get the coords of the cell
                    if move_action != 9:
                        prediction["coords"] = self.get_global_coords(
                            move_action, [row_index, col_index])
                    # If prediction is a predict move, get random coords
                    else:
                        prediction["coords"] = self.get_random_undiscovered_cell()

                    parts_predictions.append(prediction)
                    move_action += 1

        return parts_predictions

    def get_global_coords(self, move, part_coords):
        # #
        # Get coords from a part board and the selected move
        # #
        coords = self.action_list[move]
        return [coords[0] + part_coords[0], coords[1] + part_coords[1]]

    #
    # ----- Random method -----
    #

    def get_random_prediction(self):
        # #
        # Get random prediction (random move on a random part of board)
        # #
        move = randint(0, NB_ACTIONS)
        row = randint(0, self.env.nrows - WORKING_SIZE)
        col = randint(0, self.env.ncols - WORKING_SIZE)

        prediction = {
            "state": Get_board_part(self.env.get_player_board(), WORKING_SIZE, row, col).reshape(-1, NB_ACTIONS),
            "move": move,
            "part_coords": [row, col]
        }

        if move == NB_ACTIONS:
            # Pick a random undiscovered cell
            prediction["coords"] = self.get_random_undiscovered_cell()
        else:
            prediction["coords"] = self.get_global_coords(
                move, [row, col])

        return prediction

    def get_random_undiscovered_cell(self):
        # #
        # Pick a random undiscovered cell from random coords
        # #
        discovered_cell = True
        while discovered_cell:
            coords = self.get_random_coords()
            # Check if the cell is undiscovered
            if self.env.playing_board[coords[0]][coords[1]] == -1:
                discovered_cell = False

        return coords

    def get_random_coords(self):
        return [randint(0, self.env.nrows - 1), randint(0, self.env.ncols - 1)]

    #
    # ----- Model & Training -----
    #
    def load_replay_memory(self):
        if os.path.isfile(f'DDQLN/replays/{MODEL_NAME}.pkl'):
            return pickle.load(open(f'DDQLN/replays/{MODEL_NAME}.pkl', "rb"))
        return deque(maxlen=REPLAY_MAX_SIZE)

    def load_model(self):
        if os.path.isfile(f'DDQLN/models/{MODEL_NAME}'):
            return keras.models.load_model(f'DDQLN/models/{MODEL_NAME}')
        return self.init_model()

    def load_target_model(self):
        if os.path.isfile(f'DDQLN/targets/{MODEL_NAME}'):
            return keras.models.load_model(f'DDQLN/targets/{MODEL_NAME}')
        return self.init_model()

    def init_model(self):
        # Flatten
        model = tf.keras.models.Sequential()
        # Add the layers
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dense(
            NB_ACTIONS + 1, activation="linear"))

        model.compile(optimizer=keras.optimizers.Adam(lr=self.learn_rate, epsilon=1e-4),
                      loss="mse")

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done):
        # #
        # Train agent with a sample of the replay memory
        # #
        if len(self.replay_memory) < REPLAY_MIN_SIZE:
            return

        memory_sample = sample(self.replay_memory, BATCH_SIZE)

        # Predict current states
        current_states = np.array([transition[0]
                                   for transition in memory_sample])
        current_qs_list = self.model_predict(current_states).numpy()

        # Predict states after action
        new_current_states = np.array(
            [transition[3] for transition in memory_sample])
        future_qs_list = self.target_predict(new_current_states).numpy()

        X, Y = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(memory_sample):
            # Calcul new q value
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q

            else:
                new_q = reward
            # Modify the predicted state after action with the new q value
            current_qs = current_qs_list[i]
            current_qs[action] = new_q
            # Append the prediction of the current state q value and of the target q value in X and Y
            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(np.array(X), np.array(Y), batch_size=BATCH_SIZE,
                       shuffle=False)

        if done:
            self.target_update_counter += 1

        if self.target_update_counter > SWAP_TARGET:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # Decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        # Decay learn_rate
        self.learn_rate = max(LEARN_MIN, self.learn_rate * LEARN_DECAY)

    @ tf.function  # mode graph
    def model_predict(self, board):
        return self.model(board)

    @ tf.function  # mode graph
    def target_predict(self, board):
        return self.target_model(board)
