from keras.layers import LSTM, Dense, Input, InputLayer
from keras.models import Sequential

from src.const import NUMBER_OF_PLAYERS, GAME_TIME_STEP_LIMIT

INPUT_SHAPE = (GAME_TIME_STEP_LIMIT, )
LSTM_SIZE = 16


def get_model():
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(LSTM(LSTM_SIZE))
    model.add(Dense(NUMBER_OF_PLAYERS,
                    activation='softmax'))

    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model
