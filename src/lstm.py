from keras.layers import LSTM, Dense, Input, InputLayer
from keras.models import Sequential

from src.const import NUMBER_OF_PLAYERS, GAME_TIME_STEP_LIMIT

INPUT_SHAPE=(GAME_TIME_STEP_LIMIT,)


def get_model():
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(LSTM())
    model.add(Dense(NUMBER_OF_PLAYERS,
                    activation='softmax'))