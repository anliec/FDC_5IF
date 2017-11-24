from keras.layers import Conv1D, MaxPool1D, Dense, Permute, \
    Flatten, BatchNormalization, Input, Dropout
from keras.models import save_model, Model

from src.utils import *

NUMBER_OF_PLAYERS = 200

# VECTOR_CONCENTRATION_RATIO = 2.0



def get_conv_model():
    vec_in = Input(batch_shape=(None, VECTOR_DEPTH, VECTOR_SIZE))
    conv = Permute(dims=(2, 1))(vec_in)
    conv = Conv1D(filters=4,
                  kernel_size=5,
                  padding='valid',
                  activation='relu'
                  )(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=8,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=8,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=16,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(filters=16,
                  kernel_size=5,
                  padding='valid',
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = BatchNormalization()(conv)
    # conv = Permute(dims=(2, 1)))
    conv = Conv1D(filters=32,
                  kernel_size=5,
                  activation='relu')(conv)
    conv = MaxPool1D(pool_size=4)(conv)
    conv = Flatten()(conv)
    conv = BatchNormalization()(conv)

    # other_info = Input(batch_shape=(None, OTHER_INFO_SIZE,))

    # out = Concatenate()([conv, other_info])
    out = Dense(64,
                activation='relu'
                )(conv)
    out = Dropout(0.1)(out)
    out = BatchNormalization()(out)
    out = Dense(NUMBER_OF_PLAYERS,
                activation='softmax'
                )(out)
    out = Dropout(0.1)(out)

    conv_model = Model(inputs=vec_in,
                       outputs=out)

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
        reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(reader, None)
        for row in reader:
            # csv_content.append(row)
            action_list = row[1].split(',')
            player_id = row[0]
            race = action_list[0]

            game = np.zeros(shape=(VECTOR_DEPTH, VECTOR_SIZE), dtype=int)

            for i in range(1, len(action_list) - 1, 2):
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
    csv_player_game_dict = read_new_csv("data/train2.csv")

    model = get_conv_model()

    train_player_game_dict, test_player_game_dict = split_training_set(csv_player_game_dict, 0.1)

    training_input, training_input_oi, training_output, _ = csv_set_to_keras_batch(train_player_game_dict)
    test_input, test_input_oi, test_output, _ = csv_set_to_keras_batch(test_player_game_dict)

    model.fit(x=np.array(training_input),
              y=np.array(training_output),
              epochs=500,
              batch_size=16,
              validation_data=(np.array(test_input),
                               np.array(test_output)),
              verbose=1
              )

    save_model(model, "conv.knn")
