import os.path
import random
from random import randint
import time

import numpy as np
import torch

from gym.spaces import Box
from torch import nn

from gym_backgammon.envs.backgammon_env import BackgammonEnv
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, NUM_POINTS

from snowietxt_processor import roll_to_ohv, play_to_action

from decision_transformer import DecisionTransformerModel, DecisionTransformerConfig

model_dir = 'saved_models'

# Agent classes taken from https://github.com/dellalibera/td-gammon/blob/master/td_gammon/agents.py
# although I've made some substantial changes to them to work with decision transformers
class Agent:
    def __init__(self, color):
        self.color = color
        self.name = 'Agent({})'.format(COLORS[color])

    def roll(self):
        return (-randint(1, 6), -randint(1, 6)) if self.color == WHITE else (randint(1, 6), randint(1, 6))

    def choose_best_action(self, env, valid_plays, states, actions, rewards, timesteps):
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, color):
        super().__init__(color)
        self.name = 'RandomAgent({})'.format(COLORS[color])

    def choose_best_action(self, env, valid_plays, states, actions, rewards, timesteps):
        return random.choice(list(valid_plays)) if valid_plays else None

class TDModel(torch.nn.Module):
    def __init__(self):
        super(TDModel, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(198, 40),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.Linear(40, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.hidden(x)
        x = self.output(x)
        return x

class TDAgent(Agent):
    def __init__(self, color, net):
        super().__init__(color)

        if net == 'beginner':
            net = os.path.join(model_dir, 'td_gammon', 'beginner.tar')
        elif net == 'intermediate':
            net = os.path.join(model_dir, 'td_gammon', 'intermediate.tar')
        elif net == 'advanced':
            net = os.path.join(model_dir, 'td_gammon', 'advanced.tar')

        if type(net) == str:
            checkpoint = torch.load(net)
            self.net = TDModel()
            self.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net = net

        self.net.eval()

        self.name = 'TDAgent({})'.format(COLORS[color])

    @torch.no_grad()
    def choose_best_action(self, env, valid_plays, states, actions, rewards, timesteps):
        best_action = None

        if valid_plays:
            values = [0.0] * len(valid_plays)
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(valid_plays):
                observation, reward, done, info = env.step(action)
                values[i] = self.net(observation).item()

                # restore the board and other variables (undo the action)
                env.game.restore_state(state)

            # practical-issues-in-temporal-difference-learning, pag.3
            # ... the network's output P_t is an estimate of White's probability of winning from board position x_t.
            # ... the move which is selected at each time step is the move which maximizes P_t when White is to play and minimizes P_t when Black is to play.
            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(valid_plays)[best_action_index]
            env.counter = tmp_counter

        return best_action


class DTAgent(Agent):
    def __init__(self, color, model : DecisionTransformerModel, device='cuda'):
        super().__init__(color)

        self.model = model
        self.device = device

        self.model.to(device)

        # set up target_return here based on color (1 if color == WHITE, -1 if color == BLACK)
        if color == WHITE:
            self.target_return = torch.ones((1, 1), device=device)
        else:
            self.target_return = torch.ones((1, 1), device=device) * -1

    @torch.no_grad()
    def choose_best_action(self, env, valid_plays, states, actions, rewards, timesteps):
        # dt expects the plays to be 1-indexed, have bar converted, and also in the format [src1, dst1, src2, ... dst4] for each play (padded with 0,0 if there are less than 4 moves)
        valid_plays = [play_to_action(play, self.color) for play in valid_plays]

        best_play = None

        if valid_plays:

            states = states[-self.model.config.max_length:]
            actions = actions[-self.model.config.max_length:]
            timesteps = timesteps[-self.model.config.max_length:]

            seq_len = len(states)
            # skip rewards b/c we don't use it in DT
            returns_to_go = self.target_return.expand(1, seq_len, -1)

            # we'll have to pad the tensors with 0s if the sequence length is less than the max length (we do right-padding like in the paper and dataset function)
            padding = self.model.config.max_length - seq_len

            attention_mask = torch.cat([torch.zeros(padding), torch.ones(seq_len)]).unsqueeze(0).to(device=self.device, dtype=torch.long)
            states = torch.cat([torch.zeros((padding, self.model.config.state_dim)), states]).unsqueeze(0).to(self.device)
            actions = torch.cat([torch.zeros((padding, self.model.config.act_dim)), actions]).unsqueeze(0).to(self.device)
            returns_to_go = torch.cat([torch.zeros((1, padding, 1), device=self.device), returns_to_go], dim=1)
            timesteps = torch.cat([torch.ones(padding, dtype=torch.long), timesteps]).unsqueeze(0).to(self.device)

            generated_play = self.model.generate_action(states=states,
                                                   actions=actions,
                                                   valid_actions=[valid_plays], # expects plays to be batched as well
                                                   rewards=None,
                                                   returns_to_go=returns_to_go,
                                                   timesteps=timesteps,
                                                   attention_mask=attention_mask)

            generated_play = generated_play[0] # unbatch it

            # convert back to 0-indexed (and deal with the bar)
            best_play = []

            for i in range(0, len(generated_play), 2):
                src, dst = generated_play[i], generated_play[i + 1]

                if src == 0 and dst == 0: # if src and dst is 0, then it means the action that was fed the DT was padded, so we're done looking over it
                    break

                src -= 1
                dst -= 1

                if self.color == WHITE and src == NUM_POINTS:
                    src = 'bar'
                elif self.color == BLACK and src == 0:
                    src = 'bar'

                best_play.append((src, dst))

        return best_play


class DecisionTransformerBackgammonEnv(BackgammonEnv):
    def __init__(self):
        super().__init__()

        self.observation_space = Box(low=0, high=1, shape=(198 + 6 * 2,)) # 198 for Tesauro's board + 6 * 2 for a one-hot encoding of the both dice rolls
        self.action_space = Box(low=0, high=25, shape=(8,)) # actions are (src, dst) where each are from 0-25, and you can have up to 4 of them because of doubles

    def step(self, action):
        obs, reward, done, winner = super().step(action)
        if winner == BLACK:
            reward = -1

        return obs, reward, done, winner

    def roll(self):
        return random.randint(1, 6), random.randint(1, 6)

    def evaluate_agents(self, agents, n_episodes, verbose=1):
        wins = {WHITE: 0, BLACK: 0}

        act_dim = self.action_space.shape[0]

        for ep in range(n_episodes):
            agent_color, roll, observation = self.reset()
            states = torch.tensor(observation + roll_to_ohv(roll), dtype=torch.float32).unsqueeze(0)
            actions = torch.zeros((0, act_dim), dtype=torch.float32)
            rewards = torch.zeros(0, dtype=torch.float32)
            timesteps = torch.zeros((1,), dtype=torch.long)

            done = False
            agent = agents[agent_color]

            t = time.time()

            while not done:
                actions = torch.cat([actions, torch.zeros((1, act_dim), dtype=torch.float32)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1)]) # okay, we really don't use reward in the decision transformer, and especially not for backgammon but I'm keeping it around in case it becomes useful one day

                valid_actions = self.get_valid_actions(roll) # note: valid actions is a set
                action = agent.choose_best_action(self, valid_actions, states, actions, rewards, timesteps)
                new_obs, reward, done, winner = self.step(action)

                if done:
                    if winner is not None:
                        wins[agent_color] += 1
                    tot = wins[WHITE] + wins[BLACK]
                    tot = tot if tot > 0 else 1

                    if verbose:
                        print(
                            "EVAL => Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(
                                ep + 1, winner, len(actions),
                                agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                                agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))

                actions[-1] = torch.tensor(play_to_action(action, agent_color), dtype=torch.float32)
                rewards[-1] = reward
                timesteps = torch.cat([timesteps, torch.ones((1,), dtype=torch.long) * len(actions)])

                agent_color = self.get_opponent_agent()
                agent = agents[agent_color]
                roll = agent.roll()

                states = torch.cat([states, torch.tensor(new_obs + roll_to_ohv(roll), dtype=torch.float32).unsqueeze(0)], dim=0)

        return wins

if __name__ == '__main__':
    env = DecisionTransformerBackgammonEnv()
    #env.evaluate_agents({WHITE: TDAgent(WHITE, 'advanced'), BLACK: TDAgent(BLACK, 'advanced')}, 10)

    model = DecisionTransformerModel.from_pretrained('output/checkpoint-57600')

    dt_agent = DTAgent(WHITE, model)

    wins = env.evaluate_agents({WHITE: dt_agent, BLACK: TDAgent(BLACK, 'beginner')}, 100)

    print(wins)
