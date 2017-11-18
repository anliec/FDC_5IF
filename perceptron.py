from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Activation, BatchNormalization, InputLayer, Dropout

from const import *
from utils import *


def get_model():
    return_model = Sequential()
    return_model.add(InputLayer(input_shape=(NUMBER_OF_RACE + NUMBER_OF_KEYS * 2,)))
    return_model.add(Dropout(0.1))
    return_model.add(BatchNormalization())
    return_model.add(Dense(NUMBER_HIDDEN_NEURONS))
    return_model.add(Activation('sigmoid'))
    return_model.add(Dropout(0.1))
    return_model.add(BatchNormalization())
    return_model.add(Dense(NUMBER_OF_PLAYERS))
    return_model.add(Activation('softmax'))

    return_model.compile(optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])

    return_model.summary()

    return return_model


class PlayerStat:
    def __init__(self):
        self.races = [0.0, 0.0, 0.0]
        self.actions_count = defaultdict(float)
        self.game_played = 0

    def add_race(self, race):
        if race == 'Zerg':
            self.races[2] += 1.0
        elif race == 'Protoss':
            self.races[1] += 1.0
        elif race == 'Terran':
            self.races[0] += 1.0

    def add_actions(self, actions_dic):
        for action, count in actions_dic.items():
            self.actions_count[action] += count

    def increment_game_count(self):
        self.game_played += 1

    def average_stats(self):
        for race_use in self.races:
            race_use /= self.game_played
        for action, count in self.actions_count.items():
            count /= self.game_played


if __name__ == "__main__":
    train_dict_csv = read_csv('data/train.csv')
    players_dict, test_players_dict = split_training_set(train_dict_csv, VALIDATION_SPLIT)

    batch_input_list, batch_output_list, player_id_dict = csv_set_to_keras_batch(players_dict)
    test_input_list, test_output_list, _ = csv_set_to_keras_batch(test_players_dict)

    model = get_model()

    model.fit(x=np.array(batch_input_list),
              y=np.array(batch_output_list),
              batch_size=len(batch_input_list),
              epochs=FIT_EPOCHS,
              validation_data=(np.array(test_input_list),
                               np.array(test_output_list)),
              verbose=2
              )

    score = model.evaluate(x=np.array(test_input_list),
                           y=np.array(test_output_list),
                           batch_size=len(test_input_list),
                           verbose=0
                           )

    score_training = model.evaluate(x=np.array(batch_input_list),
                                    y=np.array(batch_output_list),
                                    batch_size=len(test_input_list),
                                    verbose=0
                                    )

    print("\ntraining games:", len(batch_input_list))
    print("testing games: ", len(test_input_list))

    print("\nTest scores:")
    print("\t-Loss:    ", score[0])
    print("\t-Accuracy:", score[1])

    # save_model(model=model, filepath="model.knn")

    # test_players_dict = read_csv('data/test.csv')

    # prediction = model.predict(x=np.array(test_input_list),
    #                            batch_size=len(test_input_list)
    #                            )

    # prediction_player_id = []
    # prediction_player = []
    #
    # for game_prediction in prediction:
    #     player_id = np.argmax(game_prediction)
    #     prediction_player_id.append(player_id)
    #     prediction_player.append(player_id_dict[player_id])
    #
    # print(len(prediction_player_id))
    # print(prediction_player_id)
    # print(prediction_player)
    # print(len(set(prediction_player_id)))

    stat_file = open("stats.csv", 'a')
    stat_file.write(str(NUMBER_HIDDEN_NEURONS) + ", " + str(GAME_TIME_STEP_LIMIT)
                    + ", " + str(score[1]) + ", " + str(score_training[1]) + '\n')

    # outfile = open('out.csv', 'w')
    #
    # outfile.write("row ID,battleneturl\n")
    #
    # for i, player in enumerate(prediction_player):
    #     outfile.write("Row" + str(i) + "," + player + '\n')
