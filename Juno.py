import sys
import tensorflow as tf
import numpy as np
import random

from Backgammon import Backgammon
import Preprocessor
from GNUBGClient import GNUBGClient


def create_model():
    model = tf.keras.Sequential([
        #tf.keras.layers.InputLayer(input_shape=(28, ), dtype='int32'),
        #tf.keras.layers.Conv2D(32, 1, (3,3), activation='relu'),
        #tf.keras.layers.Conv2D(32, 1, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(input_shape=(28, 6, 2)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Softmax()
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model():
    Preprocessor.create_dataset()

    train_x, train_y, test_x, test_y = Preprocessor.create_train_test()

    model = create_model()
    model.fit(train_x, train_y, epochs=6)

    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
    print('accuracy = ', test_acc)

    return model


def play_backgammon(policy):
    gnubg = GNUBGClient()

    game = Backgammon()
    player = 1
    match_point = 10

    gnubg.new_match(match_point)
    print(f'Playing {match_point} point match')

    num_games = 0
    old_p1_score = 0
    old_p2_score = 0

    while gnubg.p1_score < match_point and gnubg.p2_score < match_point:
        current_board, roll = gnubg.read_board(print_board=False)
        # roll will be None when the match is over
        if roll is not None:
            game.set_board(current_board)

            boards, moves = game.get_available_positions(roll[0], roll[1], player)

            best_board, best_move = policy.predict(boards, moves)

            #print('Current board =', game.board)
            #print('Predict board =', best_board)

            gnubg.move(best_move)

        if gnubg.p1_score > old_p1_score or gnubg.p2_score > old_p2_score:
            num_games += 1

            print(f'Score after {num_games} game(s) is P1: {gnubg.p1_score}, P2: {gnubg.p2_score}')

            old_p1_score = gnubg.p1_score
            old_p2_score = gnubg.p2_score

    winner = 1 if gnubg.p1_score > gnubg.p2_score else 2

    print(f'Player {winner} won the match')

    gnubg.close()

# use to load model
# model = tf.keras.models.load_model('juno_model')
# use to save model
# model.save('juno_model')


class NNPolicy(object):
    def __init__(self):
        self.model = tf.keras.models.load_model('juno_cnn_model')

    def predict(self, boards, moves):
        bearoff_boards = [Preprocessor.create_bearoff_board(board) for board in boards]

        scores = self.model.predict(np.asarray(bearoff_boards)) # uncomment to test including the bearoff checkers
        #scores = self.model.predict(np.asarray(boards))
        best_board = None
        best_move = None
        best_score = -1

        for board, move, score in zip(bearoff_boards, moves, scores): # uncomment to test including the bearoff checkers
        #for board, move, score in zip(boards, moves, scores):
            if score[1] > best_score:  # TODO: make it work with both players
                best_board = board
                best_move = move
                best_score = score[1]  # TODO: make it work with both players

        #print(f'Best Score = {best_score}')
        return best_board, best_move


class RandomPolicy(object):

    def predict(self, boards, moves):
        index = random.randrange(len(boards))
        return boards[index], moves[index]

def main():
    model = train_model()
    #model.save('juno_cnn_model')

    #play_backgammon(NNPolicy())
    #play_backgammon(RandomPolicy())

    return 0


if __name__ == '__main__':
    sys.exit(main())