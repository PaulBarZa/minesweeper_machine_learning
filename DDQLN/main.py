from minsweeper import Environment
from DQNAgent import *
from math import *
import random
import pickle

STATS_EVERY = 100  # calculate stats every X games
SAVE_MODEL_EVERY = 300  # save model and replay every X episodes

# Game parameters
RANDOM = True
MIN_SIZE = 5
MAX_SIZE = 20
MIN_MINES_PERCENT = 1
MAX_MINES_PERCENT = 15
# Size
ROWS = 7
COLS = 7
MINES = 4

MODEL_NAME = 'model_random.h5'


def Get_board_part(board, working_size, row_index, col_index):
    # #
    # Get part of working_size x working_size from a given board
    # #
    board_part = []

    for i in range(working_size):
        board_part.append(board[row_index + i]
                          [col_index:col_index + working_size])

    return np.array(board_part)


def Write_in_file(filename, message):
    fichier = open(filename, "a")
    fichier.write(message)
    fichier.close()


if __name__ == "__main__":

    wins_list, episode_rewards = [], []

    agent = DQNAgent(Environment(ROWS, COLS, MINES))

    Write_in_file("stats.txt",
                  f"\n - - - Starting parameters : - Size : {ROWS}x{COLS}x{MINES} (Random : {RANDOM}), Model : {MODEL_NAME}, Memory size : {len(agent.replay_memory)} - - -")
    index = 0
    for i in range(SAVE_MODEL_EVERY * 2_000):
        index += 1
        # Play on random boards
        if RANDOM:
            ROWS = random.randint(MIN_SIZE, MAX_SIZE)
            COLS = random.randint(MIN_SIZE, MAX_SIZE)
            percent = random.randint(
                MIN_MINES_PERCENT, MAX_MINES_PERCENT) / 100
            MINES = ceil(percent * (ROWS * COLS))

        env = Environment(ROWS, COLS, MINES)
        agent.set_env(env)

        episode_reward = 0

        done = False
        while not done:

            prediction, guess_move = agent.get_prediction()
            # print("Prediction")
            # print(prediction)
            # print(env.get_player_board())
            coords = prediction["coords"]
            futur_board, reward, done, is_win = env.discover_cell(
                coords[0], coords[1], guess_move)
            # print("Reward")
            # print(reward)
            part_coords = prediction["part_coords"]
            futur_state = Get_board_part(
                futur_board, WORKING_SIZE, part_coords[0], part_coords[1]).reshape(-1, NB_ACTIONS)

            episode_reward += reward

            # Add the [state(t), move, reward, state(t+1), done] to the replay memory
            agent.update_replay_memory(
                (prediction["state"][0], prediction["move"], reward, futur_state[0], done))
            # Train agent in each move
            agent.train(done)

        episode_rewards.append(episode_reward)
        wins_list.append(is_win)

        if len(agent.replay_memory) < REPLAY_MIN_SIZE:
            continue

        if not index % STATS_EVERY:
            win_rate = round(
                np.sum(wins_list[-STATS_EVERY:]) / STATS_EVERY, 2)
            med_reward = round(np.median(episode_rewards[-STATS_EVERY:]), 2)

            Write_in_file("stats.txt",
                          f"\n - Episode: {index}, Win rate : {win_rate}, Median reward : {med_reward}, Epsilon : {agent.epsilon}")

        if not index % SAVE_MODEL_EVERY:
            Write_in_file("stats.txt",
                          f"\n - - - - - Save the model, target model & replay memory for model : {MODEL_NAME} - - - - -")

            agent.model.save(f'models/{MODEL_NAME}')
            agent.target_model.save(f'targets/{MODEL_NAME}')
            with open(f'replays/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(agent.replay_memory, output)
