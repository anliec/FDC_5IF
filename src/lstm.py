from keras.layers import LSTM, Dense, Input, InputLayer
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from src.const import *
from src.utils import *

INPUT_SHAPE = (VECTOR_SIZE, VECTOR_DEPTH)
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


if __name__ == "__main__":
    csv_dict = read_csv_sequence("data/train2.csv")
    batch_input, batch_input_other_info, batch_output, player_id_to_name_dict \
        = csv_sequence_set_to_keras_batch(csv_dict)

    # create model
    model = KerasClassifier(build_fn=get_model, epochs=10, batch_size=10, verbose=0)

    # define the grid search parameters
    optimizer = ['RMSprop', 'Adam']
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

    grid_result = grid.fit(batch_input, batch_output)

