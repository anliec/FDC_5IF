from keras.layers import LSTM, Dense, Input, InputLayer
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from src.const import NUMBER_OF_PLAYERS, GAME_TIME_STEP_LIMIT
from src.utils import *

INPUT_SHAPE=(GAME_TIME_STEP_LIMIT,)


def get_model():
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(LSTM())
    model.add(Dense(NUMBER_OF_PLAYERS,
                    activation='softmax'))

if __name__ == "__main__":
    csv_dict = read_csv_sequence("data/train2.csv")
    batch_input, batch_input_other_info, batch_output, player_id_to_name_dict = csv_set_to_keras_batch(csv_dict)

    # create model
    model = KerasClassifier(build_fn=get_model, epochs=100, batch_size=10, verbose=0)

    # define the grid search parameters
    optimizer = ['RMSprop', 'Adam']
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

    grid_result = grid.fit(batch_input, batch_output)
