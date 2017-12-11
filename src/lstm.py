from keras.layers import LSTM, Dense, Input, InputLayer, Permute, BatchNormalization, \
    GRU, Conv1D, MaxPooling1D
from keras.models import Sequential, save_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from src.const import *
from src.utils import *

INPUT_SHAPE = (VECTOR_DEPTH, VECTOR_SIZE_LSTM)
LSTM_SIZE = 50


def get_model():
    lstm_model = Sequential()
    lstm_model.add(InputLayer(input_shape=INPUT_SHAPE))
    lstm_model.add(Permute((2, 1)))
    # lstm_model.add(BatchNormalization())  # removed to avoid division by 0
    lstm_model.add(Conv1D(filters=4,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    lstm_model.add(BatchNormalization())
    # lstm_model.add(MaxPooling1D(pool_size=2))
    lstm_model.add(LSTM(LSTM_SIZE,
                        dropout=0.1,
                        recurrent_dropout=0.1))
    lstm_model.add(BatchNormalization())
    lstm_model.add(Dense(NUMBER_OF_PLAYERS,
                         activation='softmax'))

    lstm_model.summary()

    lstm_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy'])
                       # metrics=['accuracy'])
    return lstm_model


if __name__ == "__main__":
    csv_dict = read_csv_sequence("data/train2.csv", VECTOR_SIZE_LSTM)
    print("parsed")
    players_dict, val_players_dict = split_training_set(csv_dict, VALIDATION_SPLIT)

    batch_input, batch_input_other_info, batch_output, player_id_to_name_dict \
        = csv_sequence_set_to_keras_batch(players_dict)
    val_batch_input, val_batch_input_other_info, val_batch_output, _ \
        = csv_sequence_set_to_keras_batch(val_players_dict)
    print("set to keras batch")

    # create model
    # model = KerasClassifier(build_fn=get_model, epochs=1, batch_size=10, verbose=0)
    model = get_model()

    # define the grid search parameters
    # optimizer = ['RMSprop', 'Adam']
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # param_grid = dict(optimizer=optimizer)
    # param_grid = dict()
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

    print("fit !")
    # grid_result = grid.fit(batch_input, batch_output)
    model.fit(x=batch_input,
              y=batch_output,
              validation_data=(val_batch_input, val_batch_output),
              epochs=500,
              batch_size=10,
              verbose=2
              )
    print("finished")

    save_model(model, "lstm.knn")

