import sys
sys.path.insert(1, "../environment")
sys.path.insert(2, "../CSP")

from environment import Environment
from csp_model import CSP_Model
from csp_solver import CSP_Solver
from agent import *
from math import *
import random
import pickle
import numpy as np
import os

GAMES = 150
STATS_EVERY = 150
SAVE_MODEL_EVERY = 500

# Prevents groups of cells greater than BOARD_PART_LEN, to avoid RAM issues with CSP model
BOARD_PART_LEN = 15

# Game parameters

# In real game, the first move is free
FREE_FIRST_MOVE = True

# Do shaping with CSP solver on training
SHAPING = True

# Random size and density
RANDOM = False
RANDOM_DENSITY = False
MIN_SIZE = 5
MAX_SIZE = 20
MIN_DENSITY_PERCENT = 5
MAX_DENSITY_PERCENT = 25

# If RANDOM is False define :
ROWS = 6
COLS = 6
MINES = 2

MODE = "CONDENSED"
MODEL_NAME = 'model_6x6x2_condensed.h5'


def write_in_file(filename, message):
    fichier = open(filename, "a")
    fichier.write(message)
    fichier.close()


def get_state(board, MODE):
    # Data normalisation
    board = board.astype(np.int8) / 8
    board = board.astype(np.float16)

    if MODE == "IMAGE":
        state = np.reshape(
            board, (ROWS, COLS, 1))

    elif MODE == "CONDENSED":

        unknown_state = []
        i = 0
        for row in board:
            unknown_state.append([])
            j = 0
            for cell in row:
                if cell == -0.125:
                    unknown_state[i].append(1)
                    board[i][j] = -1
                else:
                    unknown_state[i].append(0)
                j += 1
            i += 1

        state = np.stack((board, unknown_state), axis=2)

    return state
    

if __name__ == "__main__":
    
    training = not os.path.isfile(f'models/{MODEL_NAME}')

    wins_list, episode_rewards = [], []

    agent = DQNAgent(Environment(ROWS, COLS, MINES), MODE)
        
    write_in_file("stats.txt",
                  f"\n - - - Starting parameters : - Training : {training} - Size : {ROWS}x{COLS}x{MINES} (Random : {RANDOM}), Model : {MODEL_NAME}, Memory size : {len(agent.replay_memory)} - - -")
    index = 0
    for i in range(GAMES):
        print("Game :", i)
        index += 1
        # Play on random boards
        if RANDOM:
            ROWS = random.randint(MIN_SIZE, MAX_SIZE)
            COLS = random.randint(MIN_SIZE, MAX_SIZE)
            density = random.randint(MIN_DENSITY_PERCENT, MAX_DENSITY_PERCENT) / 100
            MINES = ceil(density * (ROWS * COLS))

        # Play on random mines density boards
        if RANDOM_DENSITY:
            density = random.randint(MIN_DENSITY_PERCENT, MAX_DENSITY_PERCENT) / 100
            MINES = ceil(density * (ROWS * COLS))

        env = Environment(ROWS, COLS, MINES)
        env.do_first_move(FREE_FIRST_MOVE)
        csp_model = CSP_Model(env, MINES)
        solver = CSP_Solver(env, csp_model, BOARD_PART_LEN)
        agent.set_env(env)

        episode_reward = 0

        done = False
        while not done:

            board = env.get_player_board()

            state = get_state(env.get_player_board(), MODE)

            move = agent.get_move(state)

            coord = env.coords_array[move]

            new_board, reward, done, is_win = env.discover_cell(
                coord[0], coord[1])
             
            if training and SHAPING:
                # Shaping
                if agent.epsilon > 0.1:
                    _, cells = solver.get_cell()
                    if coord in cells:
                        reward = 1.2

            new_state = get_state(new_board, MODE)

            # Add the [state(t), move, reward, state(t+1), done] to the replay memory
            agent.update_replay_memory(
                (state, move, reward, new_state, done))
            
            if training:        
                # Train agent in each move
                agent.train(done)

            episode_reward += reward

        episode_rewards.append(episode_reward)
        wins_list.append(is_win)

        if not index % STATS_EVERY:
            win_rate = round(
                np.sum(wins_list[-STATS_EVERY:]) / STATS_EVERY, 2)
            med_reward = round(np.median(episode_rewards[-STATS_EVERY:]), 2)

            write_in_file("stats.txt",
                          f"\n - Episode: {index}, Win rate : {win_rate}, Median reward : {med_reward}, Epsilon : {agent.epsilon}")

        if not index % SAVE_MODEL_EVERY:
            write_in_file("stats.txt",
                          f"\n - - - - - Save the model, target model & replay memory for model : {MODEL_NAME} - - - - -")

            agent.model.save(f'models/{MODEL_NAME}')
            agent.target_model.save(f'targets/{MODEL_NAME}')
            with open(f'replays/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(agent.replay_memory, output)
