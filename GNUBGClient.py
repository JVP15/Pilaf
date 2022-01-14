import os
import sys

import time
import numpy as np

import win32com.client as comclient
import win32api
import win32gui

from Backgammon import Backgammon

SLEEP_DURATION = 0.12
GNUBG_WINDOW_NAME = 'gnubg'
GNUBG_OUTPUT_FILE = 'gnubg_out.txt'
GNUBG_EXE_LOCATION = r'C:\Program Files (x86)\gnubg\gnubg-cli.exe'


class GNUBGClient(object):
    def __init__(self):
        self.window = None
        self.offset = 0

        self.p1_score = 0
        self.p2_score = 0
        self.match_score = 0

        self.connect()

    def connect(self):
        self.kill_console()  # I've found far too often that there is some error and the console window is still open after I ran the program
        print('GNUBG Client Connecting...')

        os.system(f'START /MIN run_gnubg.bat {GNUBG_WINDOW_NAME} {GNUBG_OUTPUT_FILE}')
        time.sleep(SLEEP_DURATION * 10) # need to give the shell a bit of time to load

        # HUGE thanks to user tzot for this Stack Overflow answer:
        # https://stackoverflow.com/a/136780
        # which has FINALLY allowed me to communicate with gnubg
        self.window = comclient.Dispatch("WScript.Shell")

        # consume the lines that were printed by opening up GNU Backgammon
        self.readlines()

        print('GNUBG Client Connected.')

    def read_board(self, print_board=False):
        """Reads the output of GNUBG and reconstructs the current board layout, as well as the current dice roll.
        It also handles reading the score of the match and terminating when one player wins.
        Normally, it returns board, (roll1, roll2), however if the match is over, it returns board, None
        because no dice were rolled."""

        self.update_score(self.readlines())  # consume lines and check if the score was changed

        self.execute('show board')

        lines = self.readlines()

        if print_board:
            for line in lines:
                print(line, end='')

        rolls = None

        if self.p1_score < self.match_score and self.p2_score < self.match_score:
            # Assuming that the agent is P1, these positions let us extract the roll from the board
            # TODO: make it work for both players
            roll_locations = [(5, 56), (5, 57)]

            roll_1 = int(_get_char_at(lines, roll_locations[0]))
            roll_2 = int(_get_char_at(lines, roll_locations[1]))

            rolls = (roll_1, roll_2)

        board = _get_board_from_lines(lines)

        return board, rolls

    def update_score(self, lines):
        """This function checks the input lines for any changes in the score of the current match"""
        # This is kinda fragile and I'm not entirely sure if it works in all cases
        # The numbers were gathers by printing the lines that were consumed before executing 'show board'
        if len(lines) > 0 and lines[1].startswith('P'):
            player = int(lines[1][1])
            if player == 1:
                self.p1_score += int(lines[1][-10])
            else:
                self.p2_score += int(lines[1][-10])

    def new_match(self, num_games):
        self.p1_score = 0
        self.p2_score = 0
        self.match_score = num_games

        self.execute(f'new match {num_games}')

        # consume the lines that were printed by starting a new game (this needs a little extra time)
        self.readlines(SLEEP_DURATION * 10)

    def move(self, list_of_moves):
        command = 'move'

        for move in list_of_moves:
            command += f' {move[0]} {move[1]}'

        self.execute(command)
        # consume the lines that were printed by calling 'move'
        #self.readlines()

    def execute(self, command):
        self.window.AppActivate(GNUBG_WINDOW_NAME) # may be redundant, may not be. I don't know.
        self.window.SendKeys(command + '{ENTER}')

    def readlines(self, sleep_duration=SLEEP_DURATION):
        # we want to wait a bit so that GNU Backgammon can write to the file
        time.sleep(sleep_duration)
        # while os.path.getsize(GNUBG_OUTPUT_FILE) <= self.offset:
        #    time.sleep(sleep_duration)

        with open(GNUBG_OUTPUT_FILE, 'r') as input_file:
            input_file.seek(self.offset)
            lines = input_file.readlines()
            self.offset = input_file.tell()

        return lines

    def close(self):
        # close GNU Backgammon (in the future, I may save the game or something)
        self.execute('exit')
        self.execute('y')

        # then close the window
        self.kill_console()

        print('GNUBG Client Closed.')

    def kill_console(self):
        console_window = win32gui.FindWindow(None, 'gnubg')

        if console_window != 0:
            # 0x0010 is the WM_CLOSE message
            win32api.SendMessage(console_window, 0x0010, 0, 0)

# The syntax for getting a single character in a list of strings is kinda messy, so I'll just use this instead
def _get_char_at(lines, row_col_tuple):
    return lines[row_col_tuple[0]][row_col_tuple[1]]


def _get_board_from_lines(lines):
    # These each position in the following list is the where we start reading the space
    #   given by the positions index in the list.
    # E.g., index = 0 is the position for P2's bar, index = 3 is the position of the 3rd space on the board, etc

    space_positions = [(14, 22), (4, 41), (4, 38), (4, 35), (4, 32), (4, 29), (4, 26), (4, 18), (4, 15), (4, 12), (4, 9), (4, 6), (4, 3),  # this is the top row
                       (14, 3), (14, 6), (14, 9), (14, 12), (14, 15), (14, 18), (14, 26), (14, 29), (14, 32), (14, 35), (14, 38), (14, 41), (4, 22)]  # this is the bottom row

    def read_num_checkers(start_index, increment):
        num_checkers = 0
        player = 1
        row = start_index[0]
        col = start_index[1]

        for i in range(5):
            char = _get_char_at(lines, (row + increment * i, col))

            if char == ' ':
                break
            elif char == 'X':
                num_checkers -= 1
                player = 2
            elif char == 'O':
                num_checkers += 1
            else:  # in this case, there must 6 or more checkers (GNUBG encodes them using hexidecimal)
                num_checkers = int(char, 16)
                if player == 2:
                    num_checkers = -1 * num_checkers

        return num_checkers

    board = Backgammon().board

    for i in range(26):
        # determines whether to go up or down when finding # checkers (1 means down and -1 means up)
        row_increment = 1

        # i = 0 is x's bar, which is on the bottom row, and i = 13 -> 24 are spaces on the bottom row
        if i == 0 or (13 <= i <= 24):
            row_increment = -1

        board[i] = read_num_checkers(space_positions[i], row_increment)

    return board

def main():
    gnubg = GNUBGClient()


    gnubg.new_match(1)

    gnubg.read_board()

    gnubg.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
