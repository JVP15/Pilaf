"""
This file is meant to be run from gnubg using the command gnubg -t -p collect_dataset_gnubg.py
It
"""
#import gnubg
import random
import os


CALLS_FILE = 'gnubg_calls.txt'
OUTPUT_DIR = 'games'

DIFFICULTIES = ['beginner', 'intermediate', 'advanced', 'world_class']
NUM_GAMES_PER_MATCH = 60
TOTAL_MATCHES = 2

gnubg_calls = 0 # this is how many times we've called gnubg -t -p collect_dataset_gnubg.py


def set_player_difficulty(player, difficulty):
    if difficulty == 'beginner':
        set_player_beginner(player)
    elif difficulty == 'intermediate':
        set_player_intermediate(player)
    elif difficulty == 'advanced':
        set_player_advanced(player)
    elif difficulty == 'world_class':
        set_player_world_class(player)


def set_player_beginner(player):
    gnubg.command(f'set player {player} chequer evaluation plies 0')
    gnubg.command(f'set player {player} chequer evaluation prune off')
    gnubg.command(f'set player {player} chequer evaluation noise 0.060')
    gnubg.command(f'set player {player} cube evaluation plies 0')
    gnubg.command(f'set player {player} cube evaluation prune off')
    gnubg.command(f'set player {player} cube evaluation noise 0.060')


def set_player_intermediate(player):
    gnubg.command(f'set player {player} chequer evaluation plies 0')
    gnubg.command(f'set player {player} chequer evaluation prune off')
    gnubg.command(f'set player {player} chequer evaluation noise 0.040')
    gnubg.command(f'set player {player} cube evaluation plies 0')
    gnubg.command(f'set player {player} cube evaluation prune off')
    gnubg.command(f'set player {player} cube evaluation noise 0.040')

def set_player_advanced(player):
    gnubg.command(f'set player {player} chequer evaluation plies 0')
    gnubg.command(f'set player {player} chequer evaluation prune off')
    gnubg.command(f'set player {player} chequer evaluation noise 0.015')
    gnubg.command(f'set player {player} cube evaluation plies 0')
    gnubg.command(f'set player {player} cube evaluation prune off')
    gnubg.command(f'set player {player} cube evaluation noise 0.015')

def set_player_world_class(player):
    gnubg.command(f'set player {player} chequer evaluation plies 2')
    gnubg.command(f'set player {player} chequer evaluation prune on')
    gnubg.command(f'set player {player} chequer evaluation noise 0.000')
    gnubg.command(f'set player {player} movefilter 1 0 0 8 0.160')
    gnubg.command(f'set player {player} movefilter 2 0 0 8 0.160')
    gnubg.command(f'set player {player} movefilter 3 0 0 8 0.160')
    gnubg.command(f'set player {player} movefilter 3 2 0 2 0.040')
    gnubg.command(f'set player {player} movefilter 4 0 0 8 0.160')
    gnubg.command(f'set player {player} movefilter 4 2 0 2 0.040')
    gnubg.command(f'set player {player} cube evaluation plies 2')
    gnubg.command(f'set player {player} cube evaluation prune on')
    gnubg.command(f'set player {player} cube evaluation noise 0.000')


def setup_gnubg():
    """
    This function does stuff like setting up automatic dice rolling, game playing, CPU players, etc.
    It needs to be called once every this script is ran before any games are played.
    """

    gnubg.command('set matchlength 60')
    gnubg.command('set automatic game on')
    gnubg.command('set automatic roll on')

    # I don't like to play with doubles
    gnubg.command('set automatic doubles 0')
    gnubg.command('set jacoby off')

    gnubg.command('set player 0 gnubg')
    gnubg.command('set player 1 gnubg')

    # don't print anything while playing the game
    gnubg.command('set display off')

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

def read_and_increment_file():
    global gnubg_calls

    if not os.path.exists(CALLS_FILE):
        with open(CALLS_FILE, 'w') as f:
            f.write('1')
            return 0
    else:
        with open(CALLS_FILE, 'r') as f:
            gnubg_calls = int(f.read())
        with open(CALLS_FILE, 'w') as f:
            f.write(str(gnubg_calls + 1))
        return gnubg_calls

def play_match(match_num):
    p0_difficulty = random.choice(DIFFICULTIES)
    p1_difficulty = random.choice(DIFFICULTIES)

    set_player_difficulty(0, p0_difficulty)
    set_player_difficulty(1, p1_difficulty)

    gnubg.command('new match')

    output_file = os.path.join(OUTPUT_DIR, f'{p0_difficulty}_vs_{p1_difficulty}_{gnubg_calls}_{match_num}.txt')

    gnubg.command(f'export match snowietxt {output_file}')


def play_matches():
    read_and_increment_file()
    setup_gnubg()

    for i in range(TOTAL_MATCHES):
        play_match(i)

play_matches()




