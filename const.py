import numpy as np

# set seed to be able to get same result on same run
np.random.seed(6)

NUMBER_OF_KEYS = 33
NUMBER_OF_RACE = 3

NUMBER_HIDDEN_NEURONS = 25
NUMBER_NEURON_ENCODER = 25

NUMBER_OF_PLAYERS = 71

# FIT_BATCH_SIZE = 4 # set to the len of test file
FIT_EPOCHS = 5000

VALIDATION_SPLIT = 0.1

GAME_TIME_STEP_LIMIT = 7000


def read_int_from_file(default_val, file):
    try:
        ret = int(file.readline())
    except ValueError:
        ret = default_val
    return ret


# set const value from config file
try:
    with open("config", mode='r') as config:
        NUMBER_HIDDEN_NEURONS = read_int_from_file(NUMBER_HIDDEN_NEURONS, config)
        GAME_TIME_STEP_LIMIT = read_int_from_file(GAME_TIME_STEP_LIMIT, config)
except OSError:
    pass
