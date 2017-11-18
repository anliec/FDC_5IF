from keras.models import Sequential
from keras.layers import InputLayer, BatchNormalization, Dense, Activation, Dropout

from const import *
from utils import *


def get_model():
    in_out_size = NUMBER_OF_RACE + NUMBER_OF_KEYS * 2
    return_model = Sequential()
    return_model.add(InputLayer(input_shape=(in_out_size,)))
    # return_model.add(Dropout(0.1))
    return_model.add(BatchNormalization())
    return_model.add(Dense(NUMBER_NEURON_ENCODER))
    return_model.add(Activation('sigmoid'))
    # return_model.add(Dropout(0.1))
    return_model.add(BatchNormalization())
    return_model.add(Dense(in_out_size))
    return_model.add(Activation('relu'))

    return_model.compile(optimizer='adadelta',
                         # loss='binary_crossentropy'
                         loss='mse'
                         )

    return_model.summary()

    return return_model


if __name__ == "__main__":
    train_dict_csv = read_csv('data/train.csv')
    players_dict, test_players_dict = split_training_set(train_dict_csv, VALIDATION_SPLIT)

    batch_input_list, _, player_id_dict = csv_set_to_keras_batch(players_dict)
    test_input_list, _, _ = csv_set_to_keras_batch(test_players_dict)

    model = get_model()

    model.fit(x=np.array(batch_input_list),
              y=np.array(batch_input_list),
              batch_size=len(batch_input_list),
              epochs=10000,
              validation_data=(np.array(test_input_list),
                               np.array(test_input_list)),
              verbose=2
              )

