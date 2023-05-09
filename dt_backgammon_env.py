
import random
import numpy as np
import torch

from gym.spaces import Box

from gym_backgammon.envs.backgammon_env import BackgammonEnv
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS


# Agent classes taken from https://github.com/dellalibera/td-gammon/blob/master/td_gammon/agents.py
# although I've made some substantial changes to them to work with decision transformers
class Agent:
    def __init__(self, color):
        self.color = color
        self.name = 'Agent({})'.format(COLORS[color])

    def choose_best_action(self, env, valid_actions, states, actions, rewards, timesteps):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'RandomAgent({})'.format(COLORS[color])

    def choose_best_action(self, env, valid_actions, states=None, actions=None, rewards=None, timesteps=None):

        if valid_actions is not None:
            return random.choice(actions)
        else:
            return None

class TDAgent(Agent):
    def __init__(self, color, net):
        super().__init__(color)
        self.net = net
        self.name = 'TDAgent({})'.format(COLORS[color])

    def choose_best_action(self, env, valid_actions, states=None, actions=None, rewards=None, timesteps=None):
        # we will have to extract the last state and also remove the roll from the state list

        best_action = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(actions):
                observation, reward, done, info = env.step(action)
                values[i] = self.net(observation)

                # restore the board and other variables (undo the action)
                env.game.restore_state(state)

            # practical-issues-in-temporal-difference-learning, pag.3
            # ... the network's output P_t is an estimate of White's probability of winning from board position x_t.
            # ... the move which is selected at each time step is the move which maximizes P_t when White is to play and minimizes P_t when Black is to play.
            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action


class DTAgent(Agent):
    def __init__(self, color, model):
        pass
        # set up target_return here based on color (1 if color == WHITE, -1 if color == BLACK)

    def choose_best_action(self, env, valid_actions, states, actions, rewards, timesteps):
        # we'll need to convert the valid actions from 0-indexed to 1-indexed and also remove 'bar'
        #  also handle padding the states and everything like unsqueezing n stuff
        pass


class DecisionTransformerBackgammonEnv(BackgammonEnv):
    def __init__(self):
        self.observation_space = Box(low=0, high=1, shape=(198 + 6 * 2,)) # 198 for Tesauro's board + 6 * 2 for a one-hot encoding of the both dice rolls
        self.action_space = Box(low=0, high=25, shape=(8,)) # actions are (src, dst) where each are from 0-25, and you can have up to 4 of them because of doubles

    def step(self, action):
        obs, reward, done, winner = super().step(action)
        if winner == BLACK:
            reward = -1

        return obs, reward, done, winner

    def roll(self):
        return random.randint(1, 6), random.randint(1, 6)

    def evaluate_agents(self, agents, n_episodes):
        wins = {WHITE: 0, BLACK: 0}

        for ep in n_episodes:
            agent_color, roll, observation = self.reset()
            states = None
            actions = torch.zeros((0, self.action_space.shape[0]), dtype=torch.float32)
            rewards = torch.zeros(0, dtype=torch.float32)



