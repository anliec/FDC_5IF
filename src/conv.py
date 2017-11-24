import csv
from collections import defaultdict

import numpy as np
from keras.layers import Conv1D, MaxPool1D, Dense, Permute, \
    Flatten, BatchNormalization, Input, Concatenate, Dropout
from keras.models import Sequential, save_model, Model

from src.utils import split_training_set

NUMBER_OF_PLAYERS = 200

# VECTOR_CONCENTRATION_RATIO = 2.0
VECTOR_SIZE = 10538
VECTOR_DEPTH = 10 + 1
OTHER_INFO_SIZE = 3 + 1


def get_conv_model():
    vec_in = Input(batch_shape=(None, VECTOR_DEPTH, VECTOR_SIZE))
    conv = Permute(dims=(2, 1))(vec_in)
    conv = Conv1D(filters=4,
                  kernel_size=5,
                  padding='valid',
                  activation='relu'
                  )(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=8,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=8,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=16,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=16,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    # conv = Permute(dims=(2, 1)))
    conv = Conv1D(filters=32,
                  kernel_size=5,
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = Flatten()(conv)
    conv = BatchNormalization()(conv)

    # other_info = Input(batch_shape=(None, OTHER_INFO_SIZE,))

    # out = Concatenate()([conv, other_info])
    out = Dense(64,
                activation='relu'
                )(conv)
    out = Dropout(0.1)(out)
    out = BatchNormalization()(out)
    out = Dense(NUMBER_OF_PLAYERS,
                activation='softmax'
                )(out)
    out = Dropout(0.1)(out)

    conv_model = Model(inputs=vec_in,
                       outputs=out)

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


def read_new_csv(file_name):
    players_action_dict = defaultdict(list)
    with open(file_name, 'r') as csvfile:
        number_of_line = sum(1 for _ in csvfile)  # count the number of line in the file
        csvfile.seek(0)  # set the file cursor back to file start
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line_number, row in enumerate(reader):
            action_list = row[2:]
            player_id = row[0]
            race = row[1]

            game = np.zeros(shape=(VECTOR_DEPTH, VECTOR_SIZE), dtype=int)

            current_index = 0
            for current_action in action_list:
                if current_action[0] == 't':
                    continue
                vector_index, value = action_name_to_vector_id(current_action)
                game[vector_index][current_index] = value
                current_index += 1

            # compute additional information
            relative_line_position = line_number / number_of_line
            other_info = (relative_line_position,)

            players_action_dict[player_id].append((race, game, other_info))
    return players_action_dict


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
    batch_input_other_info = []

    player_id_to_name_dict = {}
    for i, t in enumerate(csv_dict.items()):
        player_id_to_name_dict[i] = t[0]
        for race, action_vector, other_info in t[1]:
            race_list = [0, 0, 0]

            if race == 'Zerg':
                race_list[2] = 1
            elif race == 'Protoss':
                race_list[1] = 1
            elif race == 'Terran':
                race_list[0] = 1
            else:
                print("unknown race:", race)
                exit(10)
            input_array = action_vector
            input_other_info = race_list + list(other_info)
            output_array = np.zeros(shape=NUMBER_OF_PLAYERS, dtype=int)
            output_array[i] = 1

            batch_input.append(input_array)
            batch_input_other_info.append(input_other_info)
            batch_output.append(output_array)
    return batch_input, batch_input_other_info, batch_output, player_id_to_name_dict


if __name__ == "__main__":
    csv_player_game_dict = read_new_csv("data/train2.csv")

    model = get_conv_model()

    train_player_game_dict, test_player_game_dict = split_training_set(csv_player_game_dict, 0.1)

    training_input, training_input_oi, training_output, _ = csv_set_to_keras_batch(train_player_game_dict)
    test_input, test_input_oi, test_output, _ = csv_set_to_keras_batch(test_player_game_dict)

    model.fit(x=np.array(training_input),
              y=np.array(training_output),
              epochs=500,
              batch_size=16,
              validation_data=(np.array(test_input),
                               np.array(test_output)),
              verbose=1
              )

    save_model(model, "conv.knn")
