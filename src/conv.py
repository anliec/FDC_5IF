from keras.layers import Conv1D, MaxPool1D, Dense, Permute, \
    Flatten, BatchNormalization, Input, Dropout, InputLayer
from keras.models import save_model, Sequential

from src.utils import *

NUMBER_OF_PLAYERS = 200

VECTOR_CONCENTRATION_RATIO = 1.0


def get_conv_model():
    conv_model = Sequential()
    conv_model.add(InputLayer(input_shape=(VECTOR_DEPTH, VECTOR_SIZE)))
    conv_model.add(Permute(dims=(2, 1)))
    conv_model.add(Conv1D(filters=4,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'
                          ))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=8,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=8,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(BatchNormalization())
    conv_model.add(Conv1D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    conv_model.add(MaxPool1D(pool_size=4))
    # conv_model.add(BatchNormalization())
    # conv_model.add(Permute(dims=(2, 1)))
    #conv_model.add(Conv1D(filters=32,
                          # kernel_size=5,
                          # activation='relu'))
    # conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(Flatten())

    # other_info = Input(batch_shape=(None, OTHER_INFO_SIZE,))

    # conv_model.add(Concatenate()([conv, other_info])
    conv_model.add(Dense(10,
                         activation='sigmoid'
                         ))
    conv_model.add(Dropout(0.1))
    conv_model.add(BatchNormalization())
    conv_model.add(Dense(NUMBER_OF_PLAYERS,
                         activation='softmax'
                         ))
    conv_model.add(Dropout(0.2))

    # conv_model = Model(inputs=vec_in,
    #                    outputs=out)

    conv_model.summary()

    conv_model.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy'])
    return conv_model


if __name__ == "__main__":
    csv_player_game_dict = read_csv_sequence("data/train2.csv")

    model = get_conv_model()

    train_player_game_dict, test_player_game_dict = split_training_set(csv_player_game_dict, 0.01)
    train_player_game_dict = csv_player_game_dict

    training_input, _, training_output, _ = csv_sequence_set_to_keras_batch(train_player_game_dict)
    test_input, test_input_oi, test_output, _ = csv_sequence_set_to_keras_batch(test_player_game_dict)

    model.fit(x=np.array(training_input),
              y=np.array(training_output),
              epochs=1000,
              batch_size=16,
              validation_data=(np.array(test_input),
                               np.array(test_output)),
              verbose=1
              )

    save_model(model, "conv.knn")
