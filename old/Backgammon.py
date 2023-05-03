import sys
import numpy as np
from PIL import Image, ImageDraw

class Backgammon(object):
    """This class handles the logic for playing backgammon, such as whether a player can remove a checker from the board,
    or what moves a player could make given the current board and the roll of the dice. This logic is messy and confusing as heck.
    To future Jordan or anyone else looking at this code, good luck.

    Hahahaha, - Past Jordan"""

    def __init__(self):
        self.board = np.zeros(26, dtype=np.int32)
        self.setup_board()

    def setup_board(self):
        """The backgammon board is a 26 element array. Player 1 starts high and is trying to get to index 0.
        Player 2 starts low and is trying to get to index 25. In this board, 25 is the bar for P1 and 0 is the bar for P2.
        Player 1 is indicated by positive values. Player 2 is indicated by negative values."""

        self.board[1] = -2
        self.board[6] = 5
        self.board[8] = 3
        self.board[12] = -5
        self.board[13] = 5
        self.board[17] = -3
        self.board[19] = -5
        self.board[24] = 2

    def set_board(self, board: np.array):
        self.board = board.copy().astype(dtype=np.int32)

    def display_board(self):
        out = Image.new('RGB', (240, 60), (128,128,128))
        display = ImageDraw.Draw(out)

        for i in range(1, 25):
            pos = i - 1
            if self.board[i] > 0:
                display.line((5 + pos * 10, 60, 5 + pos * 10, 60 - self.board[i] * 10), width=6, fill=(255,0,0))
            elif self.board[i] < 0:
                display.line((5 + pos * 10, 60, 5 + pos * 10, 60 + self.board[i] * 10), width=6, fill=(0, 255, 255))

        display.line((235, 0, 235, self.board[25] * 10), width=6, fill=(255, 0, 0))
        display.line((5, 0, 5, -self.board[0] * 10), width=6, fill=(0, 255, 255))

        out.show()

    def move_checker(self, source, dest, player):
        self.board = Backgammon._move_checker(self.board, source, dest, player)

    @staticmethod
    def _move_checker(src_board, source, dest, player):
        """Returns a board that was created using the source board and the given move"""
        board = src_board.copy()

        # Player 2 goes in the opposite direction as player 1, so we have to change the indices for the source and dest
        if player == 2:
            source = 25 - source
            dest = 25 - dest

        # remove checker from the source index
        if player == 1:
            board[source] -= 1
        elif player == 2:
            board[source] += 1

        # if the destination is 0 or less (for P1) or 25 or more (for P2),
        # then it means that we are removing a checker from play and we can just return the board
        if dest <= 0 or dest >= 25:
            return board

        # Player 1 moves a checker
        if player == 1:
            # if we hit a checker, send it back to P2's bar and set the number of checkers in the dest to 1
            if board[dest] < 0:
                board[0] -= 1
                board[dest] = 1
            # otherwise, just increase the number of P1 checkers in the dest
            else:
                board[dest] += 1

        # Player 2 moves a checker
        elif player == 2:
            # if we hit a checker, send it back to P1's bar and set the number of checkers in the dest to -1
            if board[dest] > 0:
                board[25] += 1
                board[dest] = -1
            # otherwise, just increase the number of P2 checkers in the dest
            else:
                board[dest] -= 1

        return board

    def get_available_positions(self, roll_1, roll_2, player):
        """Generates all board configurations that are possible using the results of the roll.
        Returns two lists. The first list all possible board configurations,
        and the second list contains the list of moves taken to reach each position
        (both lists are in the same order, so the first list of moves corresponds to the first board)
        If there are no positions given the current board and the rolls, then it returns empty lists"""

        # doubles (when you roll the same number on both dice) are handled differently
        if roll_1 == roll_2:
            available_positions, available_moves = Backgammon._get_available_positions_doubles(self.board, roll_1, player)
        else:
            available_positions, available_moves = Backgammon._get_available_positions(self.board, roll_1, roll_2, player)

        return available_positions, available_moves

    @staticmethod
    def _get_available_positions(board, roll_1, roll_2, player):
        """Sometimes, it doesn't matter what order the checkers are moved in Backgammon, but in some situations
        (like when a checker is on the bar) it does matter. This is why we need to find all board configurations
        using both roll_1 and roll_2 as the first move. Handles both the situations where the order doesn't matter
        and when the order does matter, so long as roll_1 != roll_2. Returns a list containing all possible positions
        and another list containing the list of moves to reach each position (both lists are in the same order, so the
        first list of moves corresponds to the first board)"""

        # TODO: make this function more efficient by eliminating the generation of redundant board configurations
        available_positions = list()
        available_moves = list()

        # this gets us the list of possible positions using the first roll
        roll_1_positions_1st_move, roll_1_moves_1st_move = Backgammon._get_available_positions_one_roll(board, roll_1, player)

        # this gets us the list of possible positions using the second roll
        roll_2_positions_1st_move, roll_2_moves_1st_move = Backgammon._get_available_positions_one_roll(board, roll_2, player)

        # here we find all positions using roll_1 as the first move and roll_2 as the second move
        roll_2_positions_2nd_move = list()
        roll_2_moves_2nd_move = list()
        for board, move in zip(roll_1_positions_1st_move, roll_1_moves_1st_move):
            new_positions, new_moves = Backgammon._get_available_positions_one_roll(board, roll_2, player)
            roll_2_positions_2nd_move.extend(new_positions)
            roll_2_moves_2nd_move.extend(Backgammon._create_move_tuples(move, new_moves))

        # then we find all positions using roll_2 as the first move and roll_1 as the second move
        roll_1_positions_2nd_move = list()
        roll_1_moves_2nd_move = list()
        for board, move in zip(roll_2_positions_1st_move, roll_2_moves_1st_move):
            new_positions, new_moves = Backgammon._get_available_positions_one_roll(board, roll_1, player)
            roll_1_positions_2nd_move.extend(new_positions)
            roll_1_moves_2nd_move.extend(Backgammon._create_move_tuples(move, new_moves))

        # now that we've found every possible board configuration, we need to check to see which lists have legal positions

        # checks if it is possible to use roll_1 as the first move but not roll_2
        if len(roll_1_positions_1st_move) > 0 and len(roll_2_positions_1st_move) == 0:
            # checks if there are legal moves by using roll_1 and roll_2 in that order
            if len(roll_2_positions_2nd_move) > 0:
                available_positions = roll_2_positions_2nd_move
                available_moves = roll_2_moves_2nd_move
            # otherwise, just return the positions that use roll_1
            else:
                available_positions = roll_1_positions_1st_move
                available_moves = roll_1_moves_1st_move

        # check the inverse where using roll_2 first creates legal moves but using roll_1 first doesn't
        elif len(roll_2_positions_1st_move) > 0 and len(roll_1_positions_1st_move) == 0:
            # checks if there are legal moves by using roll_2 and roll_1 in that order
            if len(roll_1_positions_2nd_move) > 0:
                available_positions = roll_1_positions_2nd_move
                available_moves = roll_1_moves_2nd_move
            # otherwise, just use return the positions that use roll_2
            else:
                available_positions = roll_2_positions_1st_move
                available_moves = roll_2_moves_1st_move

        # otherwise, there are legal moves using both roll_1 and roll_2 as the first move
        else:
            # if for whatever reason, you can make a legal move with roll_1 or roll_2, but not
            # roll_1, rolL_2 or roll_2, roll_1, this handles that.
            # the rule is that if you can use either roll 1 or roll 2, you must use the higher of the two rolls
            # it also handles the case when there are no possible moves
            if len(roll_1_positions_2nd_move) == 0 and len(roll_2_positions_2nd_move) == 0:
                if roll_2 > roll_1:
                    available_positions.extend(roll_2_positions_1st_move)
                    available_moves.extend(roll_2_moves_1st_move)
                else:
                    available_positions.extend(roll_1_positions_1st_move)
                    available_moves.extend(roll_1_moves_1st_move)
            # this handles the 'normal' case where it doesn't matter in which order roll_1 and roll_2 are used
            else:
                available_positions.extend(roll_2_positions_2nd_move)
                available_positions.extend(roll_1_positions_2nd_move)
                available_moves.extend(roll_2_moves_2nd_move)
                available_moves.extend(roll_1_moves_2nd_move)

        # Unfortunately, the current method creates some duplicate board configurations.
        # We can remove them by creating a dict using the moves for each board as the key
        # Since the moves are stored as a tuple of tuples, it all works out

        position_dict = dict()

        for board, move in zip(available_positions, available_moves):
            position_dict[move] = board

        return list(position_dict.values()), list(position_dict.keys())

    @staticmethod
    def _get_available_positions_doubles(board, roll, player):
        """When you roll doubles, it means you can move up to 4 checkers, so we have to find new positions two more times.
        Since all of the numbers are the same, it doesn't matter what order they are put it, and if there are no legal
        positions given any one roll, we can ignore all of the other dice."""

        # this gets us the list of possible positions using the first roll
        roll_1_positions, roll_1_moves = Backgammon._get_available_positions_one_roll(board, roll, player)

        roll_2_positions = list()
        roll_2_moves = list()
        # since you move two pieces at a time, we have to find all of all of the ways we can use the 2nd roll
        for board, move in zip(roll_1_positions, roll_1_moves):
            new_positions, new_moves = Backgammon._get_available_positions_one_roll(board, roll, player)
            roll_2_positions.extend(new_positions)
            roll_2_moves.extend(Backgammon._create_move_tuples(move, new_moves))

        # this handles the 3rd move
        roll_3_positions = list()
        roll_3_moves = list()
        for board, move in zip(roll_2_positions, roll_2_moves):
            new_positions, new_moves = Backgammon._get_available_positions_one_roll(board, roll, player)
            roll_3_positions.extend(new_positions)
            roll_3_moves.extend(Backgammon._create_move_tuples(move, new_moves))

        roll_4_positions = list()
        roll_4_moves = list()
        # this handles the 4th move
        for board, move in zip(roll_3_positions, roll_3_moves):
            new_positions, new_moves = Backgammon._get_available_positions_one_roll(board, roll, player)
            roll_4_positions.extend(new_positions)
            roll_4_moves.extend(Backgammon._create_move_tuples(move, new_moves))

        # this checks to see which of the rolls results in at least one legal position
        if len(roll_1_positions) > 0 and len(roll_2_positions) == 0:
            available_positions = roll_1_positions
            available_moves = roll_1_moves
        # there could be a situation where the first 2 rolls result in legal positions, but the 3rd doesnt
        elif len(roll_2_positions) > 0 and len(roll_3_positions) == 0:
            available_positions = roll_2_positions
            available_moves = roll_2_moves
        # likewise, there may be a situation where the 3rd roll is usable, but the 4th isn't
        elif len(roll_3_positions) > 0 and len(roll_4_positions) == 0:
            available_positions = roll_3_positions
            available_moves = roll_3_moves
        # otherwise, we have found our possible positions
        else:
            available_positions = roll_4_positions
            available_moves = roll_4_moves

        return available_positions, available_moves

    @staticmethod
    def _get_available_positions_one_roll(board, roll, player):
        """This function returns the list of boards that could be made based on the given board and the single die roll
        as well as the moves that were taken to generate each board. Each move is stored wrapped by an iterable
        container because it makes life convenient when you dont' know whether a single turn requires 1, 2 , or 4 moves.
        """

        available_positions = list()
        moves = list()

        if player == 1:
            # for player 1, we start high and go low
            for i in range(25, 0, -1):
                if board[i] > 0 and Backgammon._can_move(board, i, i - roll, player):
                    new_board = Backgammon._move_checker(board, i, i - roll, player)
                    available_positions.append(new_board)
                    move = ((i, max(i - roll, 0)), )  # Store the move in an iterable container (see the docstring)
                    moves.append(move)

        elif player == 2:
            # for player 2, we start low and go high
            for i in range(1, 25):
                if board[i] < 0 and Backgammon._can_move(board, i, i + roll, player):
                    new_board = Backgammon._move_checker(board, i, i + roll, player)
                    available_positions.append(new_board)
                    move = (i, min(i + roll, 25))  # Store the move in an iterable container (see the docstring)
                    moves.append(move)

        return available_positions, moves

    @staticmethod
    def _create_move_tuples(original_moves, new_moves):
        """Usage: _create_move_tuples(tuple(m1,m2, ...), iterable(tuple(m3,), tuple(m4,)))
        returns [(m1,m2, ..., m3), (m1, m2, ..., m4)]
        I know it seems weird that original_moves must be a tuple, and then new_moves is an iterable collection of tuples,
        and each of those tuples probably only contain 1 move (although they can any arbitrary number of moves). I need
        to use tuples because they are hashable (which is how I remove duplicate boards). I originally had problems when
        there was only one possible move, and by making everything tuples, it all conveniently works out.
        """
        new_move_lists = list()

        for move in new_moves:
            new_move_lists.append(original_moves + move)

        return new_move_lists

    def can_remove_checkers(self, player):
        return Backgammon._can_remove_checkers(self.board, player)

    @staticmethod
    def _can_remove_checkers(board, player):
        if player == 1:
            # P1 can pull checkers off if all of their checkers are in the first 6 spaces (reminder: 0 is P2's bar)
            for num_checkers in board[7:]:
                if num_checkers > 0:
                    return False

        elif player == 2:
            # P2 can pull checkers off if all of their checkers are in the last 6 spaces (space 19 - 24, 25 is P1's bar)
            for num_checkers in board[0:19]:
                if num_checkers < 0:
                    return False

        return True

    @staticmethod
    def _can_remove_particular_checker(board, src_index, dest_index, player):
        # if we can't remove checkers at all, then we don't have to check for anything else and can just return false
        if not Backgammon._can_remove_checkers(board, player):
            return False

        # if the dest_index is 0 (for P1) or 25 (for P2),
        # then it means that the dice rolled exactly the right number required to remove it
        elif dest_index == 0 or dest_index == 25:
            return True
        # if the dest index is < 0, then we know it is P1's move, but more importantly,
        # we know that the number rolled on the die was higher than the space of the source index
        # if there are no checkers in a higher # space that the src, then we can remove that checker
        #   (e.g., we rolled a 4, but we only have checkers in space 1-3, then we can remove a checker in space 3)
        # otherwise, we cannot remove the checker
        elif dest_index < 0:
            # check all spaces in P1's home board that are greater than the src index
            # if there is a checker in any one of them, it means we cannot remove the checker from the src index
            for i in range(src_index + 1, 7):
                if board[i] > 0:
                    return False

            # if we've gotten to this point, it means that src_index is the largest space that has a checker
            # and we can remove it
            return True

        # same as above, but with P2 instead of P1
        elif dest_index > 25:
            for i in range(src_index - 1, 19, -1):
                if board[i] < 0:
                    return False

    def can_move(self, src_index, dest_index, player):
        return Backgammon._can_move(self.board, src_index, dest_index, player)

    @staticmethod
    def _can_move(board, src_index, dest_index, player):

        legal_move = False

        # before checking anything else, we need to make sure that there are no checkers on the player's bar,
        # or that the player is moving a checker from their bar onto the board
        if player == 1 and board[25] != 0 and src_index != 25:
            legal_move = False
        elif player == 2 and board[0] != 0 and src_index != 0:
            legal_move = False

        # if dest is 0 or below, then it means P1 is trying to remove a tile
        elif dest_index <= 0:
            legal_move = Backgammon._can_remove_particular_checker(board, src_index, dest_index, player)
        # if dest is 25 or above, then it means P2 is trying to remove a tile
        elif dest_index >= 25:
            legal_move = Backgammon._can_remove_particular_checker(board, src_index, dest_index, player)

        # this handles 'normal' move checking for P1
        elif player == 1:
            # P1 can move into a space if it is empty, if P1 already has a checker there, or if P2 only has one checker
            if board[dest_index] >= -1:
                legal_move = True

        # this handles 'normal' move checking for P2
        elif player == 2:
            # P2 can move into a space if it is empty, if P2 already has a checker there, or if P1 only has one checker
            if board[dest_index] <= 1:
                legal_move = True

        return legal_move

def main():

    game = Backgammon()
    boards, moves = game.get_available_positions(4, 2, player=1)
    # TODO: there seems to be a bug when checking moves for player 2
    game.move_checker(8, 4, player=1)
    src_board = game.board
    game.move_checker(6, 4, player=1)
    dest_board = game.board

    print(np.concatenate([src_board, dest_board], 0).shape)
    #print(boards)
    #print(moves)
    return 0


if __name__ == '__main__':
    sys.exit(main())

