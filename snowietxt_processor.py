
from tqdm import tqdm

from gym_backgammon.envs.backgammon import Backgammon, WHITE, BLACK, NUM_POINTS, BAR

# ================= these are util functions that are useful elsewhere (and also here) ================= #

def roll_to_ohv(roll):
    roll_ohv = [0] * 12
    # roll could be negative values (used by gym-backgammon) which is why we do abs
    roll_ohv[abs(roll[0]) - 1] = 1
    roll_ohv[abs(roll[1]) - 1 + 6] = 1

    return roll_ohv

def play_to_action(play, player):
    """
    Takes a play (list of moves like (src, dst)) and converts it to an action array [src1, dst1, src2, ..., dst4]
    The play is 0-indexed,
    """

    # we need to pad the moves with empty moves (0,0) if there are less than 4 moves
    action = [0, 0] * 4  # 4 moves

    if play is not None:
        for i in range(len(play)):
            # check if the src is bar (dst won't be bar so we don't check for that)
            if play[i][0] == BAR and player == WHITE:
                action[2 * i] = 25  # we treat white's bar as 25...
            elif play[i][0] == BAR and player == BLACK:
                action[2 * i] = 0  # so we have to treat black's bar as 0
            else:
                action[2 * i] = play[i][0] + 1  # +1 because gym_backgammon is 0 indexed, but if we treat black's bar as 0, we need to treat the board as 1-indexed

            action[2 * i + 1] = play[i][1] + 1

    return action

# ================= these are the functions used to create the dataset ================= #

def create_dataset():
    games = read_games()

    observations = list()
    actions = list()
    rewards = list()
    dones = list()

    for game in tqdm(games):
        boards = game['boards']
        bars = game['bars']
        offs = game['offs']
        rolls = game['rolls']
        players = game['players']
        moves = game['moves']

        obs_seq = list()
        action_seq = list()

        for (board, bar, off, roll, play, player) in zip(boards, bars, offs, rolls, moves, players):

            action = play_to_action(play, player)

            action_seq.append(action)
            obs = board + roll_to_ohv(roll)

            obs_seq.append(obs)

        reward = 1 if game['winner'] == WHITE else -1
        reward_seq = [0] * (len(boards) - 1) + [reward]
        done_seq = [False] * (len(boards) - 1) + [True]

        assert len(obs_seq) == len(action_seq) == len(reward_seq) == len(done_seq), f'Lengths of sequences are not equal: {len(obs_seq)}, {len(action_seq)}, {len(reward_seq)}, {len(done_seq)}'

        observations.append(obs_seq)
        actions.append(action_seq)
        rewards.append(reward_seq)
        dones.append(done_seq)

    return {'observations': observations, 'actions': actions, 'rewards': rewards, 'dones': dones}

def read_games():
    games = []
    for i in range(0, 10):
        #filename = f'games\\match{i}.txt' # used grandmaster
        filename = f'games\\game{i}'  # not sure what I used for this
        #print('Opening', filename)

        with open(filename, 'r') as match_file:
            # skip header
            match_file.readline()
            match_file.readline()

            while True:
                game = parse_game(match_file)
                if game is None:
                    break # we've reached the last game in the file
                games.append(game)
                # skip the whitespace between games
                match_file.readline()

    print('Number of games', len(games))

    return games

def parse_game(in_file):
    line = in_file.readline()

    if 'Game' not in line: # this lets us know if we've reached the end of the file, so return None to indicate that there are no more games
        return None

    game_board = Backgammon()
    boards = list()
    bars = list()
    offs = list()
    rolls = list()
    moves = list()
    players = list()

    # skip player scores
    in_file.readline()

    line = in_file.readline()

    # If there are 4 spaces, then it means that the line contains the winner of the game
    while not line.startswith('    '):
        move_str_1, move_str_2, p1_roll, p2_roll = parse_line(line)

        if move_str_1 is not None: # if it is None, then it means p1 didn't have a turn (e.g. its the first turn and p2 went first)
            #boards.append(game_board.board.copy())
            boards.append(game_board.get_board_features(current_player=WHITE)) # this gets us the state that the board would be in if it were white's turn
            bars.append(game_board.bar.copy())
            offs.append(game_board.off.copy())

            move = apply_move(game_board, player=WHITE, move_str=move_str_1)
            moves.append(move)
            rolls.append(p1_roll)
            players.append(WHITE)

        if move_str_2 is not None: # if it is None, then it means p2 didn't have a turn (game probably ended on p1's turn)
            #boards.append(game_board.board.copy())
            boards.append(game_board.get_board_features(current_player=BLACK)) # this gets us the state that the board would be in if it were black's turn
            bars.append(game_board.bar.copy())
            offs.append(game_board.off.copy())

            move = apply_move(game_board, player=BLACK, move_str=move_str_2) # then we apply the move after getting the state so it is ready for the next player's turn
            moves.append(move)
            rolls.append(p2_roll)
            players.append(BLACK)

        line = in_file.readline()

    # now, line contains the winner of the game and how many points they won.
    # TODO: implement different number of points

    # If P1 wins, we give it a score of 1
    winner = WHITE
    # when P1 wins, there are 6 spaces before the number of points, but there are 34 spaces when P2 wins
    # so if there are 7 spaces at the start of the line, we know that P2 won, and we say that the value is 1
    if line.startswith('       '):
        winner = BLACK

    game = {'winner': winner,
            'players' : players,
            'boards': boards,
            'bars': bars,
            'offs': offs,
            'moves': moves,
            'rolls': rolls}

    return game


def parse_line(line):
    colon_index = line.index(':')
    move_str_1 = None
    move_str_2 = None
    p1_roll = None
    p2_roll = None

    # this handles the case where P2 goes first and P1 doesn't have a move on the first line
    if colon_index == 35:
        # we're adding 2 to the colon index because the moves start 2 spaces after the colon
        move_str_2 = line[colon_index + 2:].strip()

        p2_roll_1 = int(line[colon_index - 2:colon_index - 1])
        p2_roll_2 = int(line[colon_index - 1:colon_index])
        p2_roll = (p2_roll_1, p2_roll_2)

    else:
        # normal case
        try:
            colon_index_2 = line.index(':', colon_index + 1)

            # we're adding 2 to the colon index because the moves start 2 spaces after the colon
            # we're subtracting 3 from the second colon index because the roll is stored before the colon
            # and we don't want to read it in the move string
            move_str_1 = line[colon_index + 2:colon_index_2 - 3].strip()

            p1_roll_1 = int(line[colon_index - 2:colon_index - 1])
            p1_roll_2 = int(line[colon_index - 1:colon_index])
            p1_roll = (p1_roll_1, p1_roll_2)

            move_str_2 = line[colon_index_2 + 2:].strip()

            p2_roll_1 = int(line[colon_index_2 - 2:colon_index_2 - 1])
            p2_roll_2 = int(line[colon_index_2 - 1:colon_index_2])
            p2_roll = (p2_roll_1, p2_roll_2)

        # this handles the case where P1 ends the game and P2 doesn't have a move on the last line
        except ValueError:
            # we're adding 2 to the colon index because the moves start 2 spaces after the colon
            move_str_1 = line[colon_index + 2:].strip()

            p1_roll_1 = int(line[colon_index - 2:colon_index - 1])
            p1_roll_2 = int(line[colon_index - 1:colon_index])
            p1_roll = (p1_roll_1, p1_roll_2)

    return move_str_1, move_str_2, p1_roll, p2_roll

def transform_move(player, move):
    # snowie works with 1-indexed moves, but gym-backgammon 0 indexes them, so we have to deal with that
    # also snowie records relative moves (i.e. p2 and p1 move from high to low) but gym-backgammon works with absolute moves (p1 goes high to low, p2 goes low to high)
    # gym-backgammon also uses a special BAR character
    src_point = int(move[0])
    dst_point = int(move[1])

    if player == WHITE:
        src_point = src_point - 1
        dst_point = dst_point - 1
    elif player == BLACK:
        src_point = NUM_POINTS - src_point
        dst_point = NUM_POINTS - dst_point

    if int(move[0]) == NUM_POINTS + 1:
        src_point = BAR

    return (src_point, dst_point)



def apply_move(backgammon, player, move_str):
    """This function takes a backgammon object, the current player, and a single string containing all the moves
    by that player (it expects that this string has also been stripped of heading and trailing whitespace).
    If the string is empty, it will return an empty move (, ) and not change the board."""

    if len(move_str) == 0: # this happens if p1 or p2 had a turn and rolled, but can't move any pieces
        return tuple()

    # we don't need to have a * (indicating that a piece was taken) so we want to remove it
    moves_no_star = move_str.replace('*', '')

    moves_split = moves_no_star.split(' ')

    list_of_moves = list(map(lambda x: x.split('/'), moves_split))

    # convert the strings into a list of move tuples (it'll look like [(1,2), (3,4)])

    play = [transform_move(player, move) for move in list_of_moves]
    backgammon.execute_play(player, play)

    return play


if __name__ == '__main__':
    games = read_games()
    print(games[0]['boards'][0], games[0]['winner'])

    # calculate how many boards we have in the dataset
    num_boards = 0
    num_games = len(games)
    for game in games:
        num_boards += len(game['boards'])

    print('Number of boards in dataset: ', num_boards)
    print('Number of games in dataset: ', num_games)

    dataset = create_dataset()

    print(dataset['observations'][0][0])
    print(dataset['actions'][0][0])
    print(dataset['rewards'][0][0])



