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
    conv_model.add(BatchNormalization())
    # conv_model.add(Permute(dims=(2, 1)))
    conv_model.add(Conv1D(filters=32,
                          kernel_size=5,
                          activation='sigmoid'))
    conv_model.add(MaxPool1D(pool_size=4))
    conv_model.add(Flatten())

    # other_info = Input(batch_shape=(None, OTHER_INFO_SIZE,))

    # conv_model.add(Concatenate()([conv, other_info])
    conv_model.add(Dense(64,
                         activation='relu'
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


def read_csv(file_name):
    players_game_dict = defaultdict(list)
    collision = np.zeros(shape=(VECTOR_DEPTH,))
    number_of_zero = 0
    number_of_game = 0
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            action_list = row[2:]
            player_id = row[0]
            race = row[1]

            game = np.zeros(shape=(VECTOR_DEPTH, VECTOR_SIZE), dtype=int)

            i = 0
            for current_action in action_list:
                index, value = action_name_to_vector_id(current_action)
                if game[index][current_timestep] != 0:
                    collision[index] += 1
            for i in range(1, len(action_list) - 1):
                current_action = action_list[i]
                current_timestep = int(int(action_list[i + 1]) * VECTOR_CONCENTRATION_RATIO)
                # action_id = action_name_to_id(current_action)
                index, value = action_name_to_vector_id(current_action)
                if game[index][current_timestep] != 0:
                    collision[index] += 1
                game[index][current_timestep] = value

            number_of_game += 1
            number_of_zero += VECTOR_SIZE * VECTOR_DEPTH - np.count_nonzero(game)

            players_game_dict[player_id].append((race, game))
    print("number of collision:", np.sum(collision))
    print("collision per game :", np.sum(collision) / number_of_game)
    print("number of game:     ", number_of_game)
    print("sparsity:           ", number_of_zero * 100 / (VECTOR_SIZE * VECTOR_DEPTH * number_of_game))
    print("collision detail")
    for i, v in enumerate(collision):
        print("hotkey" + str(i), v)
    return players_game_dict


if __name__ == "__main__":
    csv_player_game_dict = read_csv_sequence("data/train2.csv")

    model = get_conv_model()

    # train_player_game_dict, test_player_game_dict = split_training_set(csv_player_game_dict, 0.1)
    train_player_game_dict = csv_player_game_dict

    training_input, _, training_output, _ = csv_sequence_set_to_keras_batch(train_player_game_dict)
    # test_input, test_input_oi, test_output, _ = csv_sequence_set_to_keras_batch(test_player_game_dict)

    model.fit(x=np.array(training_input),
              y=np.array(training_output),
              epochs=500,
              batch_size=16,
              # validation_data=(np.array(test_input),
              #                  np.array(test_output)),
              verbose=1
              )

    save_model(model, "conv.knn")
