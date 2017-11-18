from keras.models import Sequential, save_model, load_model
from keras.layers import InputLayer, Conv1D, MaxPool1D, Dense, Permute, Flatten, BatchNormalization
import csv
from collections import defaultdict
import numpy as np

from utils import split_training_set

NUMBER_OF_PLAYERS = 71

VECTOR_CONCENTRATION_RATIO = 1.0
VECTOR_SIZE = int(6720 * VECTOR_CONCENTRATION_RATIO)
VECTOR_DEPTH = 10 + 1


def get_conv_model():
    conv_model = Sequential()
    conv_model.add(InputLayer(input_shape=(VECTOR_DEPTH, VECTOR_SIZE)))
    conv_model.add(Permute(dims=(2, 1)))
    conv_model.add(Conv1D(filters=8,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=8,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=8,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Permute(dims=(2, 1)))
    conv_model.add(Dense(1,
                         activation='relu'))
    conv_model.add(Flatten())
    conv_model.add(Dense(NUMBER_OF_PLAYERS,
                         activation='softmax'))
    conv_model.summary()

    conv_model.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy'])
    return conv_model


def action_name_to_id(action_name):
    if type(action_name) == str:
        if action_name[:6] == 'hotkey':
            hotkey_id = int(action_name[6])
            hotkey_action = int(action_name[7])
            return 10 * hotkey_action + hotkey_id + 1 + 3
        elif action_name == 's':
            return 1
        elif action_name == 'sBase' or action_name == 'Base':
            return 3
        elif action_name == 'sMineral' or action_name == 'SingleMineral':
            return 2
        else:
            return 0


def action_name_to_vector_id(action_name):
    if type(action_name) == str:
        if action_name[:6] == 'hotkey':
            hotkey_id = int(action_name[6])
            hotkey_action = int(action_name[7])
            if hotkey_action > 0:
                index = 2
            else:
                index = 1
            return hotkey_id, index
        elif action_name == 's':
            return 1, 1
        elif action_name == 'sBase' or action_name == 'Base':
            return 1, 3
        elif action_name == 'sMineral' or action_name == 'SingleMineral':
            return 1, 2
        else:
            return 0, 0


def read_csv(file_name):
    players_game_dict = defaultdict(list)
    collision = np.zeros(shape=(VECTOR_DEPTH,))
    number_of_zero = 0
    number_of_game = 0
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(reader, None)
        for row in reader:
            # csv_content.append(row)
            action_list = row[1].split(',')
            player_id = row[0]
            race = action_list[0]

            game = np.zeros(shape=(VECTOR_DEPTH, VECTOR_SIZE), dtype=int)

            for i in range(1, len(action_list) - 1, 2):
                current_action = action_list[i]
                current_timestep = int(int(action_list[i + 1]) * VECTOR_CONCENTRATION_RATIO)
                # action_id = action_name_to_id(current_action)
                index, value = action_name_to_vector_id(current_action)
                if game[index][current_timestep] != 0:
                    collision[index] += 1
                game[index][current_timestep] = value

            number_of_game += 1
            number_of_zero += VECTOR_SIZE * VECTOR_DEPTH - np.count_nonzero(game)

            players_game_dict[player_id].append((race, game))
    print("number of collision:", np.sum(collision))
    print("collision per game :", np.sum(collision) / number_of_game)
    print("number of game:     ", number_of_game)
    print("sparsity:           ", number_of_zero * 100 / (VECTOR_SIZE * VECTOR_DEPTH * number_of_game))
    print("collision detail")
    for i, v in enumerate(collision):
        print("hotkey" + str(i), v)
    return players_game_dict


def csv_set_to_keras_batch(csv_dict):
    batch_input = []
    batch_output = []

    player_id_to_name_dict = {}
    for i, t in enumerate(csv_dict.items()):
        player_id_to_name_dict[i] = t[0]
        for race, action_vector in t[1]:
            # race_list = [0, 0, 0]
            #
            # if race == 'Zerg':
            #     race_list[2] = 1
            # elif race == 'Protoss':
            #     race_list[1] = 1
            # elif race == 'Terran':
            #     race_list[0] = 1
            # else:
            #     print("unknown race:", race)
            #     exit(10)
            input_array = action_vector
            output_array = np.zeros(shape=NUMBER_OF_PLAYERS, dtype=int)
            output_array[i] = 1

            batch_input.append(input_array)
            batch_output.append(output_array)
    return batch_input, batch_output, player_id_to_name_dict


if __name__ == "__main__":
    csv_player_game_dict = read_csv("data/train.csv")
    model = get_conv_model()

    train_player_game_dict, test_player_game_dict = split_training_set(csv_player_game_dict, 0.1)

    training_input, training_output, _ = csv_set_to_keras_batch(train_player_game_dict)
    test_input, test_output, _ = csv_set_to_keras_batch(test_player_game_dict)

    model.fit(x=np.array(training_input),
              y=np.array(training_output),
              epochs=100,
              batch_size=64,
              validation_data=(np.array(test_input),
                               np.array(test_output)),
              verbose=1
              )

    save_model(model, "conv.knn")
