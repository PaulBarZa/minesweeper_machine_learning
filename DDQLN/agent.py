import tensorflow as tf
import numpy as np
import os.path
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, concatenate
from random import sample
from collections import deque
from ddqln_main import*


# Exploration settings
epsilon = 0.24
EPSILON_MIN = 0.01

# Deep Q-learning model parameters
learn_rate = 0.00004
# LEARN_MIN = 0.0001
DISCOUNT = 0
DECAY = 0.99975

# Memory & Target
SWAP_TARGET = 10
BATCH_SIZE = 400
REPLAY_MIN_SIZE = 1_000
REPLAY_MAX_SIZE = 70_000


class DQNAgent(object):
    def __init__(self, env, mode):
        # Environment
        self.env = env
        self.mode = mode
        # Deep Q-learning Parameters
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        # Init model and target model
        self.model = self.load_model()
        self.target_model = self.load_target_model()
        self.replay_memory = self.load_replay_memory()
        self.target_update_counter = 0

    def set_env(self, env):
        self.env = env

    def get_move(self, state):
        # #
        # Get a move prediction
        # #

        if self.mode == "CONDENSED":
            unknown_value = -1
            state_dim = 2

            board_first_channel = []
            i = 0
            for row in state:
                board_first_channel.append([])
                for cell in row:
                    board_first_channel[i].append(cell[0])
                i += 1

            board = np.array(board_first_channel).reshape(1, self.env.ntiles)

        elif self.mode == "IMAGE":
            unknown_value = -0.125
            state_dim = 1
            board = state.reshape(1, self.env.ntiles)

        unsolved = [i for i, x in enumerate(board[0]) if x == unknown_value]

        # Random selection b/w exploiration or exploitation
        rand = np.random.random()
        # Explore
        if rand < self.epsilon:
            move = np.random.choice(unsolved)

       # Exploit
        else:
            moves = self.model.predict(np.reshape(
                state, (1, self.env.nrows, self.env.ncols, state_dim)))

            # set already clicked tiles to min value
            moves[board != unknown_value] = np.min(moves)
            move = np.argmax(moves)

        return move

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
        return self.model

    def init_model(self):

        if self.mode == "IMAGE":

            input = Input(shape=(self.env.nrows, self.env.ncols, 1))

            cnv2d_1 = Conv2D(128, (3, 3), activation='relu',
                             padding='same')(input)
            cnv2d_2 = Conv2D(128, (3, 3), activation='relu',
                             padding='same')(cnv2d_1)

            b1_add = concatenate([cnv2d_1, cnv2d_2])

            cnv2d_3 = Conv2D(128, (3, 3), activation='relu',
                             padding='same')(b1_add)
            cnv2d_4 = Conv2D(128, (3, 3), activation='relu',
                             padding='same')(cnv2d_3)

            b2_add = concatenate([cnv2d_3, cnv2d_4])

            flatten = Flatten()(b2_add)

            dense_1 = Dense(512, activation='relu')(flatten)
            dense_2 = Dense(512, activation='relu')(dense_1)
            output = Dense(self.env.ntiles, activation='linear')(dense_2)

            model = Model(input, output)

        elif self.mode == "CONDENSED":

            model = tf.keras.Sequential([
                Conv2D(18, (5, 5), activation=tf.nn.relu, padding='same',
                       input_shape=(self.env.nrows, self.env.ncols, 2)),
                Conv2D(
                    36, (3, 3), activation=tf.nn.relu, padding='same'),
                Flatten(),
                Dense(288, activation='relu'),
                Dense(220, activation='relu'),
                Dense(200, activation='relu'),
                Dense(self.env.ntiles, activation='softmax')])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learn_rate, epsilon=1e-4),
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

        for i, (current_state, action, reward, _, done) in enumerate(memory_sample):
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
        self.epsilon = max(EPSILON_MIN, self.epsilon * DECAY)

        # Decay learn_rate
        # self.learn_rate = max(LEARN_MIN, self.learn_rate * DECAY)

    @ tf.function  # mode graph
    def model_predict(self, board):
        return self.model(board)

    @ tf.function  # mode graph
    def target_predict(self, board):
        return self.target_model(board)
