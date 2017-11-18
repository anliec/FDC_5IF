import csv
from collections import defaultdict, OrderedDict
from const import *


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

            other_info = (line_number / number_of_line,)

            players_action_dict[player_id].append((race, ordered_action_dict, ordered_action_first_time_dict, other_info))
    return players_action_dict


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
                           'hotkey80': 0, 'SingleMineral': 0, 'hotkey02': 0, 'hotkey82': 0, 'hotkey92': 0, 'hotkey91': 0,
                           'hotkey01': 0, 'hotkey41': 0, 'hotkey21': 0, 'hotkey71': 0, 'hotkey81': 0, 'hotkey61': 0,
                           'hotkey11': 0, 'hotkey51': 0, 'hotkey31': 0}
            current_timestep = 0
            for i in action_list:
                current_action = i
                if current_action[0] == 't':
                    current_timestep += 5
                    continue
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

            other_info = (line_number / number_of_line,)

            players_action_dict[player_id].append((race, ordered_action_dict, ordered_action_first_time_dict, other_info))
    return players_action_dict


def split_training_set(source_dict, test_to_train_ratio=0.1):
    train_dict = {}
    test_dict = {}
    for player_name, player_game in source_dict.items():
        number_of_games = len(player_game)
        split_index = int(number_of_games - number_of_games * test_to_train_ratio)
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
        for race, action_dict, first_time_dict in t[1]:
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

            input_array = np.array([race_id]
                                   + list(action_dict.values())
                                   + list(first_time_dict.values())
                                   + list(other_info)
                                   , dtype=int)
            output_string = t[0]

            batch_input.append(input_array)
            batch_output.append(output_string)
    return batch_input, batch_output

