import sys
import random
import numpy as np
from Backgammon import Backgammon

"""Yeah... this is some horrible code that I cobbled together. It is ugly and fragile. 
Future Jordan or future collaborators, good luck."""

dataset = list()


def create_dataset():
    for i in range(1, 11):
        filename = f'games\\games{i}.txt'
        print('Opening', filename)

        with open(filename, 'r') as match_file:
            # skip header
            match_file.readline()
            match_file.readline()

            end_of_file = False

            while not end_of_file:
                end_of_file = parse_game(match_file)
                # skip the whitespace between games
                match_file.readline()

    random.shuffle(dataset)
    print('Number of board positions in dataset =', len(dataset))

def create_train_test():
    if len(dataset) == 0:
        create_dataset()

    train_size = int(len(dataset) * .8)

    train_x = np.asarray([x[0] for x in dataset[0:train_size]])
    train_y = np.asarray([x[1] for x in dataset[0:train_size]])

    test_x = np.asarray([x[0] for x in dataset[train_size:]])
    test_y = np.asarray([x[1] for x in dataset[train_size:]])

    return train_x, train_y, test_x, test_y

def parse_game(in_file):
    line = in_file.readline()

    if 'Game' not in line: # this lets us know if we've reached the end of the file
        return True

    game_board = Backgammon()
    boards = list()

    # skip player scores
    in_file.readline()

    line = in_file.readline()

    # If there are 4 spaces, then it means that the line contains the winner of the game
    while not line.startswith('    '):
        move_str_1, move_str_2 = parse_line(line)

        if move_str_1 != '':
            apply_move(game_board, player=1, move_str=move_str_1)
            boards.append(game_board.board)

        if move_str_2 != '':
            apply_move(game_board, player=2, move_str=move_str_2)
            boards.append(game_board.board)

        line = in_file.readline()

    # now, line contains the winner of the game and how many points they won.
    # TODO: implement different number of points

    # If P1 wins, we give it a score of 1
    winner = 1
    # when P1 wins, there are 6 spaces before the number of points, but there are 34 spaces when P2 wins
    # so if there are 7 spaces at the start of the line, we know that P2 won, and we say that the value is -1
    if line.startswith('       '):
        winner = 0

    for board in boards:
        dataset.append((board, winner))

    # since we've gone through the entire game, we haven't reached the end of the file
    return False


def parse_line(line):
    colon_index = line.index(':')
    move_str_1 = ''
    move_str_2 = ''

    # this handles the case where P2 goes first and P1 doesn't have a move on the first line
    if colon_index == 35:
        # we're adding 2 to the colon index because the moves start 2 spaces after the colon
        move_str_2 = line[colon_index + 2:].strip()
    else:
        # normal case
        try:
            colon_index_2 = line.index(':', colon_index + 1)

            # we're adding 2 to the colon index because the moves start 2 spaces after the colon
            # we're subtracting 3 from the second colon index because there are 2 unnecessary numbers before the colon
            move_str_1 = line[colon_index + 2:colon_index_2 - 3].strip()

            move_str_2 = line[colon_index_2 + 2:].strip()

        # this handles the case where P1 ends the game and P2 doesn't have a move on the last line
        except ValueError:
            # we're adding 2 to the colon index because the moves start 2 spaces after the colon
            move_str_1 = line[colon_index + 2:].strip()

    return move_str_1, move_str_2


def apply_move(backgammon, player, move_str):
    """This function takes a backgammon object, the current player, and a single string containing all of the moves
    by that player (it expects that this string has also been stripped of heading and trailing whitespace)"""

    # we don't need to have a * (indicating that a piece was taken) so we want to remove it
    moves_no_star = move_str.replace('*', '')

    moves_split = moves_no_star.split(' ')

    list_of_moves = list(map(lambda x: x.split('/'), moves_split))

    for move in list_of_moves:
        backgammon.move_checker(int(move[0]), int(move[1]), player)
        #print(f'P{player} moving from {move[0]} to {move[1]}')

def main():
    create_dataset()
    return 0


if __name__ == '__main__':
    sys.exit(main())
