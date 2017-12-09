from keras.layers import LSTM, Dense, Input, InputLayer, Permute, BatchNormalization, \
    GRU, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from collections import defaultdict
from src.const import *
import csv

INPUT_SHAPE = (VECTOR_DEPTH, VECTOR_SIZE_LSTM)
LSTM_SIZE = 100


def get_model():
    lstm_model = Sequential()
    lstm_model.add(InputLayer(input_shape=INPUT_SHAPE))
    lstm_model.add(Permute((2, 1)))
    lstm_model.add(BatchNormalization())
    lstm_model.add(Conv1D(filters=4,
                          kernel_size=5,
                          padding='valid',
                          activation='relu'))
    lstm_model.add(BatchNormalization())
    lstm_model.add(MaxPooling1D(pool_size=2))
    lstm_model.add(LSTM(LSTM_SIZE,
                        dropout=0.2,
                        recurrent_dropout=0.2))
    lstm_model.add(BatchNormalization())
    lstm_model.add(Dense(NUMBER_OF_PLAYERS,
                         activation='softmax'))

    lstm_model.summary()

    lstm_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy'])
                       # metrics=['accuracy'])
    return lstm_model


def read_new_csv(file_name):
    players_action_dict = defaultdict(list)
    action_dict = {}
    action_max_id = 0
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line_number, row in enumerate(reader):
            player_id = row[0]

            game_list = []

            for action in row[1:]:
                if action in action_dict:
                    action_id = action_dict[action]
                else:
                    action_id = action_max_id
                    action_dict[action] = action_max_id
                    action_max_id += 1
                game_list.append(action_id)

            # game_string = ' '.join(row[1:])

            players_action_dict[player_id].append(game_list)
    return players_action_dict, action_max_id


def split_training_set(source_dict, test_to_train_ratio=0.1):
    train_dict = {}
    test_dict = {}
    for player_name, player_game in source_dict.items():
        number_of_games = len(player_game)
        split_index = number_of_games - int(number_of_games * test_to_train_ratio) - 2
        train_game = player_game[0:split_index]
        test_game = player_game[split_index:-1]
        train_dict[player_name] = train_game
        test_dict[player_name] = test_game
    return train_dict, test_dict


def csv_set_to_keras_batch(csv_dict):
    batch_input = []
    batch_output = []

    player_id_to_name_dict = {}
    for i, t in enumerate(csv_dict.items()):
        player_id_to_name_dict[i] = t[0]
        for string in t[1]:
            output_array = np.zeros(shape=NUMBER_OF_PLAYERS, dtype=int)
            output_array[i] = 1

            batch_input.append(string)
            batch_output.append(output_array)
    return batch_input, batch_output, player_id_to_name_dict


if __name__ == "__main__":
    top_words = 1000
    value, top_words = read_new_csv("data/train_long.csv")
    print(top_words)
    train, test = split_training_set(value)

    X_train, y_train, _ = csv_set_to_keras_batch(train)
    X_test, y_test, _ = csv_set_to_keras_batch(test)
    # truncate and pad input sequences
    max_review_length = 1000
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(200, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    print(model.summary())
    exit(0)
    model.fit(np.array(X_train),
              np.array(y_train),
              nb_epoch=3,
              batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
