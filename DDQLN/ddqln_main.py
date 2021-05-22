import sys
sys.path.insert(1, "environment")
sys.path.insert(2, "CSP")

from environment import Environment
from cspinit import CSP_Model
from csp_main import get_cell
from DQNAgent import *
from math import *
import random
import pickle
import numpy as np

STATS_EVERY = 500  # calculate stats every X games
SAVE_MODEL_EVERY = 1000  # save model and replay every X episodes

# Game parameters
RANDOM = False
MIN_SIZE = 5
MAX_SIZE = 20
MIN_MINES_PERCENT = 1
MAX_MINES_PERCENT = 15
# Size
ROWS = 6
COLS = 6
MINES = 2

MODEL_NAME = 'model_6x6x2_2.1.h5'


def Write_in_file(filename, message):
    fichier = open(filename, "a")
    fichier.write(message)
    fichier.close()


if __name__ == "__main__":

    wins_list, episode_rewards = [], []

    agent = DQNAgent(Environment(ROWS, COLS, MINES))

    Write_in_file("DDQLN/stats.txt",
                  f"\n - - - Starting parameters : - Size : {ROWS}x{COLS}x{MINES} (Random : {RANDOM}), Model : {MODEL_NAME}, Memory size : {len(agent.replay_memory)} - - -")
    index = 0
    for i in range(SAVE_MODEL_EVERY * 25):
        index += 1
        # Play on random boards
        if RANDOM:
            ROWS = random.randint(MIN_SIZE, MAX_SIZE)
            COLS = random.randint(MIN_SIZE, MAX_SIZE)
            percent = random.randint(
                MIN_MINES_PERCENT, MAX_MINES_PERCENT) / 100
            MINES = ceil(percent * (ROWS * COLS))

        env = Environment(ROWS, COLS, MINES)

        csp_model = CSP_Model(env)

        agent.set_env(env)

        nb_move = 0
        episode_reward = 0

        done = False
        while not done:
            nb_move += 1

            board = env.get_player_board()  # np array 2 dims
            # print(board)
            state = np.reshape(
                board, (ROWS, COLS, 1))
            # Data normalisation
            state = state.astype(np.int8) / 8
            state = state.astype(np.float16)

            move = agent.get_move(state)
            # print("agent move", move)
            find_cell, cells = get_cell(env, csp_model)
            # print("CSP coords", cells)
            coords = env.coords_array[move]
            # print("agent coords", coords)
            new_board, reward, done, is_win = env.discover_cell(
                coords[0], coords[1])
            # print("Reward de base", reward)
            # Shaping
            if find_cell and nb_move > 1:
                if coords in cells:
                    reward = 0.9

            # input("Wait")
            new_state = np.reshape(
                new_board, (ROWS, COLS, 1)).astype(np.float16)
            # Data normalisation
            new_state = new_state.astype(np.int8) / 8
            new_state = new_state.astype(np.float16)

            episode_reward += reward

            # Add the [state(t), move, reward, state(t+1), done] to the replay memory
            agent.update_replay_memory(
                (state, move, reward, new_state, done))
            # Train agent in each move
            agent.train(done)

        # If loose first moove
        if nb_move == 1:
            index -= 1
            continue

        episode_rewards.append(episode_reward)
        wins_list.append(is_win)

        if len(agent.replay_memory) < REPLAY_MIN_SIZE:
            continue

        if not index % STATS_EVERY:
            win_rate = round(
                np.sum(wins_list[-STATS_EVERY:]) / STATS_EVERY, 2)
            med_reward = round(np.median(episode_rewards[-STATS_EVERY:]), 2)

            Write_in_file("DDQLN/stats.txt",
                          f"\n - Episode: {index}, Win rate : {win_rate}, Median reward : {med_reward}, Epsilon : {agent.epsilon}")

        if not index % SAVE_MODEL_EVERY:
            Write_in_file("DDQLN/stats.txt",
                          f"\n - - - - - Save the model, target model & replay memory for model : {MODEL_NAME} - - - - -")

            agent.model.save(f'DDQLN/models/{MODEL_NAME}')
            agent.target_model.save(f'DDQLN/targets/{MODEL_NAME}')
            with open(f'DDQLN/replays/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(agent.replay_memory, output)