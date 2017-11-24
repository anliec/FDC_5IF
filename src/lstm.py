from keras.layers import LSTM, Dense, Input, InputLayer, Permute
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from src.const import *
from src.utils import *

INPUT_SHAPE = (VECTOR_DEPTH, VECTOR_SIZE_LSTM)
LSTM_SIZE = 16


def get_model():
    lstm_model = Sequential()
    lstm_model.add(InputLayer(input_shape=INPUT_SHAPE))
    lstm_model.add(Permute((2, 1)))
    lstm_model.add(LSTM(LSTM_SIZE))
    lstm_model.add(Dense(NUMBER_OF_PLAYERS,
                         activation='softmax'))

    lstm_model.summary()

    lstm_model.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       # metrics=['categorical_accuracy'])
                       metrics=['accuracy'])
    return lstm_model


if __name__ == "__main__":
    csv_dict = read_csv_sequence("data/train2.csv", VECTOR_SIZE_LSTM)
    print("parsed")
    batch_input, batch_input_other_info, batch_output, player_id_to_name_dict \
        = csv_sequence_set_to_keras_batch(csv_dict)
    print("set to keras batch")
    # create model
    model = KerasClassifier(build_fn=get_model, epochs=1, batch_size=10, verbose=0)

    # define the grid search parameters
    # optimizer = ['RMSprop', 'Adam']
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # param_grid = dict(optimizer=optimizer)
    param_grid = dict()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

    print("fit !")
    grid_result = grid.fit(batch_input, batch_output)
    print("finished")
