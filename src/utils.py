import csv
from collections import defaultdict, OrderedDict
from keras.models import load_model

from src.const import *


def read_csv(file_name):
    players_action_dict = defaultdict(list)
    with open(file_name, 'r') as csvfile:
        number_of_line = sum(1 for _ in csvfile)  # count the number of line in the file
        csvfile.seek(0)  # set the file cursor back to file start
        reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(reader, None)
        for line_number, row in enumerate(reader):
            # csv_content.append(row)
            action_list = row[1].split(',')
            player_id = row[0]
            race = action_list[0]
            action_first_time_dict = {'s': GAME_TIME_STEP_LIMIT, 'hotkey50': GAME_TIME_STEP_LIMIT,
                                      'hotkey40': GAME_TIME_STEP_LIMIT, 'hotkey52': GAME_TIME_STEP_LIMIT,
                                      'hotkey42': GAME_TIME_STEP_LIMIT, 'hotkey10': GAME_TIME_STEP_LIMIT,
                                      'hotkey12': GAME_TIME_STEP_LIMIT, 'hotkey20': GAME_TIME_STEP_LIMIT,
                                      'hotkey22': GAME_TIME_STEP_LIMIT, 'hotkey30': GAME_TIME_STEP_LIMIT,
                                      'hotkey60': GAME_TIME_STEP_LIMIT,
                                      'hotkey62': GAME_TIME_STEP_LIMIT,
                                      'hotkey32': GAME_TIME_STEP_LIMIT, 'sBase': GAME_TIME_STEP_LIMIT,
                                      'hotkey70': GAME_TIME_STEP_LIMIT, 'hotkey72': GAME_TIME_STEP_LIMIT,
                                      'hotkey00': GAME_TIME_STEP_LIMIT,
                                      'hotkey90': GAME_TIME_STEP_LIMIT,
                                      'hotkey80': GAME_TIME_STEP_LIMIT, 'sMineral': GAME_TIME_STEP_LIMIT,
                                      'hotkey02': GAME_TIME_STEP_LIMIT, 'hotkey82': GAME_TIME_STEP_LIMIT,
                                      'hotkey92': GAME_TIME_STEP_LIMIT,
                                      'hotkey91': GAME_TIME_STEP_LIMIT,
                                      'hotkey01': GAME_TIME_STEP_LIMIT, 'hotkey41': GAME_TIME_STEP_LIMIT,
                                      'hotkey21': GAME_TIME_STEP_LIMIT, 'hotkey71': GAME_TIME_STEP_LIMIT,
                                      'hotkey81': GAME_TIME_STEP_LIMIT,
                                      'hotkey61': GAME_TIME_STEP_LIMIT,
                                      'hotkey11': GAME_TIME_STEP_LIMIT, 'hotkey51': GAME_TIME_STEP_LIMIT,
                                      'hotkey31': GAME_TIME_STEP_LIMIT}
            action_dict = {'s': 0, 'hotkey50': 0, 'hotkey40': 0, 'hotkey52': 0, 'hotkey42': 0, 'hotkey10': 0,
                           'hotkey12': 0, 'hotkey20': 0, 'hotkey22': 0, 'hotkey30': 0, 'hotkey60': 0, 'hotkey62': 0,
                           'hotkey32': 0, 'sBase': 0, 'hotkey70': 0, 'hotkey72': 0, 'hotkey00': 0, 'hotkey90': 0,
                           'hotkey80': 0, 'sMineral': 0, 'hotkey02': 0, 'hotkey82': 0, 'hotkey92': 0, 'hotkey91': 0,
                           'hotkey01': 0, 'hotkey41': 0, 'hotkey21': 0, 'hotkey71': 0, 'hotkey81': 0, 'hotkey61': 0,
                           'hotkey11': 0, 'hotkey51': 0, 'hotkey31': 0}
            current_timestep = 0
            for i in range(1, len(action_list) - 1, 2):
                current_action = action_list[i]
                current_timestep = int(action_list[i + 1])
                if current_timestep > GAME_TIME_STEP_LIMIT:
                    break
                # if current_action not in action_dict:
                #     action_dict[current_action] = 0
                action_dict[current_action] += 1
                if action_dict[current_action] == 1:
                    action_first_time_dict[current_action] = current_timestep
                # possible_action_dict[current_action] = 1

            # if player_id not in players_dict:
            #     players_dict[player_id] = []

            ordered_action_dict = OrderedDict(sorted(action_dict.items()))
            ordered_action_first_time_dict = OrderedDict(sorted(action_first_time_dict.items()))

            # compute additional information
            relative_line_position = line_number / number_of_line
            apm = len(action_list) / (current_timestep + 1)  # + 1 to prevent division by 0
            other_info = (relative_line_position, apm,)

            players_action_dict[player_id].append((race, ordered_action_dict, ordered_action_first_time_dict, other_info))
    return players_action_dict


def read_new_csv(file_name, read_all=False, time_limit=GAME_TIME_STEP_LIMIT):
    players_action_dict = defaultdict(list)
    with open(file_name, 'r') as csvfile:
        number_of_line = sum(1 for _ in csvfile)  # count the number of line in the file
        csvfile.seek(0)  # set the file cursor back to file start
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line_number, row in enumerate(reader):
            action_list = row[2:]
            player_id = row[0]
            race = row[1]
            action_first_time_dict = {'s': time_limit, 'hotkey50': time_limit,
                                      'hotkey40': time_limit, 'hotkey52': time_limit,
                                      'hotkey42': time_limit, 'hotkey10': time_limit,
                                      'hotkey12': time_limit, 'hotkey20': time_limit,
                                      'hotkey22': time_limit, 'hotkey30': time_limit,
                                      'hotkey60': time_limit,
                                      'hotkey62': time_limit,
                                      'hotkey32': time_limit, 'Base': time_limit,
                                      'hotkey70': time_limit, 'hotkey72': time_limit,
                                      'hotkey00': time_limit,
                                      'hotkey90': time_limit,
                                      'hotkey80': time_limit, 'SingleMineral': time_limit,
                                      'hotkey02': time_limit, 'hotkey82': time_limit,
                                      'hotkey92': time_limit,
                                      'hotkey91': time_limit,
                                      'hotkey01': time_limit, 'hotkey41': time_limit,
                                      'hotkey21': time_limit, 'hotkey71': time_limit,
                                      'hotkey81': time_limit,
                                      'hotkey61': time_limit,
                                      'hotkey11': time_limit, 'hotkey51': time_limit,
                                      'hotkey31': time_limit}
            action_dict = {'s': 0, 'hotkey50': 0, 'hotkey40': 0, 'hotkey52': 0, 'hotkey42': 0, 'hotkey10': 0,
                           'hotkey12': 0, 'hotkey20': 0, 'hotkey22': 0, 'hotkey30': 0, 'hotkey60': 0, 'hotkey62': 0,
                           'hotkey32': 0, 'Base': 0, 'hotkey70': 0, 'hotkey72': 0, 'hotkey00': 0, 'hotkey90': 0,
                           'hotkey80': 0, 'SingleMineral': 0, 'hotkey02': 0, 'hotkey82': 0, 'hotkey92': 0, 'hotkey91': 0,
                           'hotkey01': 0, 'hotkey41': 0, 'hotkey21': 0, 'hotkey71': 0, 'hotkey81': 0, 'hotkey61': 0,
                           'hotkey11': 0, 'hotkey51': 0, 'hotkey31': 0}
            current_timestep = 0
            max_ap5s = 0
            ap5s = 0
            for i in action_list:
                current_action = i
                if current_action[0] == 't':
                    current_timestep += 5
                    max_ap5s = max_ap5s if max_ap5s > ap5s else ap5s
                    ap5s = 0
                    continue
                if current_timestep > time_limit:
                    break
                action_dict[current_action] += 1
                ap5s += 1
                if action_dict[current_action] == 1:
                    action_first_time_dict[current_action] = current_timestep

            # order action to be sure that every game as them in the same order
            ordered_action_dict = OrderedDict(sorted(action_dict.items()))
            ordered_action_first_time_dict = OrderedDict(sorted(action_first_time_dict.items()))

            # compute additional information
            relative_line_position = line_number / number_of_line
            apm = len(action_list) / (current_timestep + 1)  # + 1 to prevent division by 0
            other_info = (relative_line_position, apm, max_ap5s)

            if current_timestep < 60 and not read_all:
                print("discarded line %i, game too short: %is" % (line_number, current_timestep))
            else:
                players_action_dict[player_id].append((race, ordered_action_dict, ordered_action_first_time_dict, other_info))
    return players_action_dict


def split_training_set(source_dict, test_to_train_ratio=0.1):
    train_dict = {}
    test_dict = {}
    for player_name, player_game in source_dict.items():
        number_of_games = len(player_game)
        split_index = number_of_games - int(number_of_games * test_to_train_ratio) - 2
        train_game = player_game[0:split_index]
        test_game = player_game[split_index:-1]
        train_dict[player_name] = train_game
        test_dict[player_name] = test_game
    return train_dict, test_dict


def csv_set_to_keras_batch(csv_dict):
    batch_input = []
    batch_output = []

    player_id_to_name_dict = {}
    for i, t in enumerate(csv_dict.items()):
        player_id_to_name_dict[i] = t[0]
        for race, action_dict, first_time_dict, other_info in t[1]:
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

            input_array = np.array(race_list
                                   + list(action_dict.values())
                                   + list(first_time_dict.values())
                                   + list(other_info)
                                   , dtype=int)
            output_array = np.zeros(shape=NUMBER_OF_PLAYERS, dtype=int)

            output_array[i] = 1

            batch_input.append(input_array)
            batch_output.append(output_array)
    return batch_input, batch_output, player_id_to_name_dict


def csv_set_to_sklearn_batch(csv_dict):
    batch_input = []
    batch_output = []

    for i, t in enumerate(csv_dict.items()):
        for race, action_dict, first_time_dict, other_info in t[1]:
            race_id = 0

            if race == 'Zerg':
                race_id = 3
            elif race == 'Protoss':
                race_id = 1
            elif race == 'Terran':
                race_id = 2
            else:
                print("unknown race:", race)
                exit(10)

            input_list = [race_id] + list(action_dict.values()) \
                                   + list(first_time_dict.values()) \
                                   + list(other_info)
            input_array = np.array(input_list
                                   , dtype=int)
            output_string = t[0]

            batch_input.append(input_array)
            batch_output.append(output_string)
    return batch_input, batch_output


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


def read_csv_sequence(file_name, lenght=VECTOR_SIZE):
    players_action_dict = defaultdict(list)
    with open(file_name, 'r') as csvfile:
        number_of_line = sum(1 for _ in csvfile)  # count the number of line in the file
        csvfile.seek(0)  # set the file cursor back to file start
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line_number, row in enumerate(reader):
            action_list = row[2:]
            player_id = row[0]
            race = row[1]

            game = np.zeros(shape=(VECTOR_DEPTH, lenght), dtype=int)

            current_index = 0
            for current_action in action_list:
                if current_action[0] == 't':
                    continue
                vector_index, value = action_name_to_vector_id(current_action)
                game[vector_index][current_index] = value
                current_index += 1
                if current_index >= lenght:
                    break

            # compute additional information
            relative_line_position = line_number / number_of_line
            other_info = (relative_line_position,)

            players_action_dict[player_id].append((race, game, other_info))
    return players_action_dict


def csv_sequence_set_to_keras_batch(csv_dict):
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
    return np.array(batch_input), np.array(batch_input_other_info), \
           np.array(batch_output), np.array(player_id_to_name_dict)


def expended_read_new_csv(file_name, read_all=False):
    # load model trained by conv.py
    conv_model = load_model("conv.knn")
    # remove last layer (the classifier is not what we look for, we want the features)
    conv_model.pop()  # drop dropout layer
    conv_model.pop()  # drop last Dense layer
    conv_model.summary()
    players_action_dict = defaultdict(list)
    with open(file_name, 'r') as csvfile:
        number_of_line = sum(1 for _ in csvfile)  # count the number of line in the file
        csvfile.seek(0)  # set the file cursor back to file start
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line_number, row in enumerate(reader):
            action_list = row[2:]
            player_id = row[0]
            race = row[1]
            action_first_time_dict = {'s': GAME_TIME_STEP_LIMIT, 'hotkey50': GAME_TIME_STEP_LIMIT,
                                      'hotkey40': GAME_TIME_STEP_LIMIT, 'hotkey52': GAME_TIME_STEP_LIMIT,
                                      'hotkey42': GAME_TIME_STEP_LIMIT, 'hotkey10': GAME_TIME_STEP_LIMIT,
                                      'hotkey12': GAME_TIME_STEP_LIMIT, 'hotkey20': GAME_TIME_STEP_LIMIT,
                                      'hotkey22': GAME_TIME_STEP_LIMIT, 'hotkey30': GAME_TIME_STEP_LIMIT,
                                      'hotkey60': GAME_TIME_STEP_LIMIT,
                                      'hotkey62': GAME_TIME_STEP_LIMIT,
                                      'hotkey32': GAME_TIME_STEP_LIMIT, 'Base': GAME_TIME_STEP_LIMIT,
                                      'hotkey70': GAME_TIME_STEP_LIMIT, 'hotkey72': GAME_TIME_STEP_LIMIT,
                                      'hotkey00': GAME_TIME_STEP_LIMIT,
                                      'hotkey90': GAME_TIME_STEP_LIMIT,
                                      'hotkey80': GAME_TIME_STEP_LIMIT, 'SingleMineral': GAME_TIME_STEP_LIMIT,
                                      'hotkey02': GAME_TIME_STEP_LIMIT, 'hotkey82': GAME_TIME_STEP_LIMIT,
                                      'hotkey92': GAME_TIME_STEP_LIMIT,
                                      'hotkey91': GAME_TIME_STEP_LIMIT,
                                      'hotkey01': GAME_TIME_STEP_LIMIT, 'hotkey41': GAME_TIME_STEP_LIMIT,
                                      'hotkey21': GAME_TIME_STEP_LIMIT, 'hotkey71': GAME_TIME_STEP_LIMIT,
                                      'hotkey81': GAME_TIME_STEP_LIMIT,
                                      'hotkey61': GAME_TIME_STEP_LIMIT,
                                      'hotkey11': GAME_TIME_STEP_LIMIT, 'hotkey51': GAME_TIME_STEP_LIMIT,
                                      'hotkey31': GAME_TIME_STEP_LIMIT}
            action_dict = {'s': 0, 'hotkey50': 0, 'hotkey40': 0, 'hotkey52': 0, 'hotkey42': 0, 'hotkey10': 0,
                           'hotkey12': 0, 'hotkey20': 0, 'hotkey22': 0, 'hotkey30': 0, 'hotkey60': 0, 'hotkey62': 0,
                           'hotkey32': 0, 'Base': 0, 'hotkey70': 0, 'hotkey72': 0, 'hotkey00': 0, 'hotkey90': 0,
                           'hotkey80': 0, 'SingleMineral': 0, 'hotkey02': 0, 'hotkey82': 0, 'hotkey92': 0,
                           'hotkey91': 0,
                           'hotkey01': 0, 'hotkey41': 0, 'hotkey21': 0, 'hotkey71': 0, 'hotkey81': 0, 'hotkey61': 0,
                           'hotkey11': 0, 'hotkey51': 0, 'hotkey31': 0}

            game = np.zeros(shape=(1, VECTOR_DEPTH, VECTOR_SIZE), dtype=int)
            current_timestep = 0
            max_ap5s = 0
            ap5s = 0
            current_index = 0
            for i in action_list:
                break_ask = 0
                current_action = i
                if current_action[0] == 't':
                    current_timestep += 5
                    max_ap5s = max(ap5s, max_ap5s)
                    ap5s = 0
                    continue
                if current_timestep > GAME_TIME_STEP_LIMIT:
                    break_ask += 1
                else:
                    action_dict[current_action] += 1
                    ap5s += 1
                    if action_dict[current_action] == 1:
                        action_first_time_dict[current_action] = current_timestep
                if current_index > VECTOR_SIZE:
                    break_ask += 1
                else:
                    vector_index, value = action_name_to_vector_id(current_action)
                    game[0][vector_index][current_index] = value
                    current_index += 1
                if break_ask == 2:
                    break

            conv_feature = conv_model.predict(x=game, batch_size=1, verbose=0)

            # order action to be sure that every game as them in the same order
            ordered_action_dict = OrderedDict(sorted(action_dict.items()))
            ordered_action_first_time_dict = OrderedDict(sorted(action_first_time_dict.items()))

            # compute additional information
            relative_line_position = line_number / number_of_line
            apm = len(action_list) / (current_timestep + 1)  # + 1 to prevent division by 0
            other_info = (relative_line_position, apm, max_ap5s) + tuple(conv_feature[0])

            if current_timestep < 60 and not read_all:
                print("discarded line %i, game too short: %is" % (line_number, current_timestep))
            else:
                players_action_dict[player_id].append(
                    (race, ordered_action_dict, ordered_action_first_time_dict, other_info))
    return players_action_dict
