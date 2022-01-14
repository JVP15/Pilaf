import sys
import tensorflow as tf
import numpy as np
import random

from Backgammon import Backgammon
import Preprocessor
from GNUBGClient import GNUBGClient


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(26,), dtype='int32'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='relu'),
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model():
    Preprocessor.create_dataset()

    train_x, train_y, test_x, test_y = Preprocessor.create_train_test()

    model = create_model()
    model.fit(train_x, train_y, epochs=10)

    test_loss, test_acc = model.evaluate(train_x, train_y, verbose=2)
    print('accuracy = ', test_acc)

    return model


def play_backgammon(policy):
    gnubg = GNUBGClient()

    game = Backgammon()
    player = 1
    match_point = 10
    gnubg.new_match(match_point)
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

            #print(f'Best Score = {best_score}')
            #print('Current board =', game.board)
            #print('Predict board =', best_board)

            gnubg.move(best_move)

        if gnubg.p1_score > old_p1_score or gnubg.p2_score > old_p2_score:
            num_games += 1

            print(f'Score after {num_games} game(s) is P1: {gnubg.p1_score}, P2: {gnubg.p2_score}')

            old_p1_score = gnubg.p1_score
            old_p2_score = gnubg.p2_score

    gnubg.close()

# use to load model
# model = tf.keras.models.load_model('pilaf_model')
# use to save model
# model.save('pilaf_model')


class NNPolicy(object):
    model = tf.keras.models.load_model('pilaf_model')

    def predict(self, boards, moves):
        scores = self.model.predict(np.asarray(boards))
        best_board = None
        best_move = None
        best_score = -1

        for board, move, score in zip(boards, moves, scores):
            if score[0] > best_score:  # TODO: make it work with both players
                best_board = board
                best_move = move
                best_score = score[0]  # TODO: make it work with both players

        return best_board, best_move


class RandomPolicy(object):

    def predict(self, boards, moves):
        index = random.randrange(len(boards))
        return boards[index], moves[index]

def main():
    #play_backgammon(NNPolicy())
    play_backgammon(RandomPolicy())

    return 0


if __name__ == '__main__':
    sys.exit(main())