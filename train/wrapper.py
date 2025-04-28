import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from coinche.gym.env import GymCoinche

import pyspiel
from open_spiel.python.algorithms import cfr, deep_cfr, nfsp
from open_spiel.python import rl_environment

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel

'''from open_spiel.python import register_game

register_game.register_game(
    "coinche",  # short name
    lambda params=None: CoincheOpenSpiel(params)
)'''

class CoincheOpenSpiel(pyspiel.Game):
    def __init__(self, params=None):
        game_type = pyspiel.GameType(
            short_name="coinche",
            long_name="Coinche Card Game",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.GENERAL_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=4,
            min_num_players=4,
            provides_information_state_string=False,
            provides_information_state_tensor=True,
            provides_observation_string=False,
            provides_observation_tensor=True,
            parameter_specification={},
            default_loadable=True,
            provides_factored_observation_string=False,
        )

        game_info = pyspiel.GameInfo(
            num_distinct_actions=44, #32,
            max_chance_outcomes=0,
            num_players=4,
            min_utility=-180,
            max_utility=180,
            utility_sum=0,
            max_game_length=37 + 4 * 8, #32 * 8,
        )

        super().__init__(game_type, game_info, params or {})


    def new_initial_state(self):
        return CoincheState(self)

    def observation_tensor_size(self):
        return 37 + 32 + 32 + 32 + 2 + 5 #98 

    def action_tensor_size(self):
        return 44

    def num_distinct_actions(self):
        return 44

    def max_game_length(self):
        return 37 + 4 * 8
    
    def num_players(self):
        return 4

class CoincheState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.env = GymCoinche()
        self.game = game
        self._observation, _ = self.env.reset()
        self._is_terminal = False
        #print(self.env)
        self._current_player = self.current_player()
        #self.env.current_trick_rotation[0]#self.env.current_player

    def current_player(self):
        #print("Player : ", self.env._get_current_player().index)
        return self.env._get_current_player().index#self._current_player

    def legal_actions(self, player=None):
        #NEED TO FIX NAME
        return self.env._legal_action()

    def apply_action(self, action):
        obs, reward, terminated, _, _ = self.env.step(action)
        self._observation = obs
        self._is_terminal = terminated
        #print(" terminated : ", terminated)
        self._current_player = self.current_player() if not terminated else pyspiel.PlayerId.TERMINAL

    def is_terminal(self):
        return self._is_terminal

    def rewards(self, player, action):
        return self.env._get_reward(self, player, action)
        #self.env._get_reward(self, player, action)
        #return self.env.returns()

    def returns(self):
        #for player in players:
        #    reward[i]
        return self.env._get_return()
        #return self.env.returns()

    def observation_tensor(self, player=None):
        obs = self.env._get_current_observation()
        return np.array(obs, dtype=np.float32)

    def information_state_tensor(self, player=None):
        return self.observation_tensor()

    def action_mask(self):
        mask = np.zeros(self.game.num_distinct_actions(), dtype=np.int8)
        mask[self.legal_actions()] = 1
        return mask

    def __str__(self):
        return f"CoincheState(player={self._current_player}, obs={self._observation})"
    
    def clone(self):
        cloned = CoincheState(self.game)
        cloned.env = self.env.copy()  # you MUST make sure your GymCoinche env has a .copy() method!
        cloned._observation = np.copy(self._observation)
        cloned._is_terminal = self._is_terminal
        cloned._current_player = self._current_player
        return cloned

# Assume CoincheOpenSpiel and CoincheState are defined elsewhere and should not be modified.

Transition = namedtuple('Transition', ['obs', 'action', 'advantage', 'player'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def make_mlp(input_dim, output_dim, hidden_sizes):
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)

def print_mlp(model):
    for idx, layer in enumerate(model):
    # some layers (ReLU, Pool) have no weights
        if hasattr(layer, 'weight'):
            print(f"Layer {idx} [{layer.__class__.__name__}] weights:")
            print(layer.weight.data)             # tensor of weights
        if hasattr(layer, 'bias') and layer.bias is not None:
            print(f"Layer {idx} [{layer.__class__.__name__}] bias:")
            print(layer.bias.data)               # tensor of biases
        print('—' * 40)

class DeepCFRWrapper:
    def __init__(self, config: dict, game_instance=None):
        """
        config: a dict provided by user (unaltered), containing keys like:
          - 'params': parameters for CoincheOpenSpiel constructor
          - 'total_timesteps': number of iterations to run
          - 'log_every': logging interval
          - 'batch_size': batch size for training
          - 'adv_hidden', 'adv_lr', 'adv_mem_size'
          - 'pol_hidden', 'pol_lr', 'pol_mem_size'
        game_instance: optional initialized CoincheOpenSpiel
        """

        self.policy_player_input = True
        # store full config
        self.user_config = config

        # default hyperparameters
        defaults = {
            'adv_hidden': [64, 64], 'adv_lr': 1e-3, 'adv_mem_size': 100_000,
            'pol_hidden': [64, 64], 'pol_lr': 1e-3, 'pol_mem_size': 100_000,
            'batch_size': 256, 'num_iterations': 100, 'log_interval': 10,
            'num_traversals': 5,
        }

        # map user config keys to our internal settings
        # for iterations, use 'total_timesteps' if present
        defaults['num_iterations'] = config.get('total_timesteps', defaults['num_iterations'])
        defaults['log_interval'] = config.get('log_every', defaults['log_interval'])
        defaults['batch_size'] = config.get('batch_size', defaults['batch_size'])

        # advantage network settings
        self.adv_hidden = config.get('adv_hidden', defaults['adv_hidden'])
        self.adv_lr = config.get('adv_lr', defaults['adv_lr'])
        self.adv_mem_size = config.get('adv_mem_size', defaults['adv_mem_size'])
        # policy network settings
        self.pol_hidden = config.get('pol_hidden', defaults['pol_hidden'])
        self.pol_lr = config.get('pol_lr', defaults['pol_lr'])
        self.pol_mem_size = config.get('pol_mem_size', defaults['pol_mem_size'])

        # training loop settings
        self.batch_size = defaults['batch_size']
        self.num_iterations = defaults['num_iterations']
        self.log_interval = defaults['log_interval']
        self.num_traversals = config.get('num_traversals', defaults['num_traversals'])
        # initialize game
        params = config.get('params', {})
        self.game = CoincheOpenSpiel(params=params)

        # derive dimensions
        self.num_players = self.game.num_players()
        # observation dim: user can override via config['obs_dim'], else from game
        self.obs_dim = config.get('obs_dim', self.game.observation_tensor_size())
        self.num_actions = self.game.num_distinct_actions()

        # build networks
        self.adv_nets = [make_mlp(self.obs_dim, self.num_actions, self.adv_hidden)
                         for _ in range(self.num_players)]
        self.adv_opts = [optim.Adam(net.parameters(), lr=self.adv_lr)
                         for net in self.adv_nets]

        self.policy_net = make_mlp(self.policy_player_input+self.obs_dim, self.num_actions, self.pol_hidden)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=self.pol_lr)

        # memories
        self.adv_memory = [ReplayBuffer(self.adv_mem_size) for _ in range(self.num_players)]
        self.pol_memory = ReplayBuffer(self.pol_mem_size)

    def learn(self, total_timesteps, callback=None, use_masking=False):
        for it in range(total_timesteps):
            #print(f"Iteration {it+1}/{total_timesteps}...")
            # initialize adv networks
            self.adv_nets = [
                make_mlp(self.obs_dim, self.num_actions, self.adv_hidden)
                for _ in range(self.num_players)
            ]
            self.adv_opts = [
                optim.Adam(net.parameters(), lr=self.adv_lr)
                for net in self.adv_nets
            ]
            self.adv_memory = [
                ReplayBuffer(self.adv_mem_size)
                for _ in range(self.num_players)
            ]
            # collect adv samples
            for player in range(self.num_players):
                for traversal in range(self.num_traversals):
                    state = self.game.new_initial_state()
                    self._traverse(state, player)
                self._optimize_adv(player)
                # policy update
            self._optimize_policy()

            # logging
            if (it + 1) % self.log_interval == 0:
                adv_sizes = [len(buf) for buf in self.adv_memory]
                print(f"[DCFR] Iter {it+1}/{self.num_iterations}, adv_sizes={adv_sizes}, pol_size={len(self.pol_memory)}")
                print(self.evaluate_policy(self.policy_net, num_episodes=50))
                #print(f"[DCFR] Policy: {print_mlp(self.policy_net)}")
                #for i, net in enumerate(self.adv_nets):
                    #print(f"[DCFR] Adv Net {i}: {print_mlp(net)}")


    def _masked_softmax(self, logits, legal):
        mask = torch.zeros_like(logits)
        mask[legal] = 1.0
        unnorm = torch.exp(logits) * mask
        return (unnorm / unnorm.sum()).detach().cpu().numpy()
    
    def _sample_chance(self, state):
        outcomes = state.chance_outcomes()
        acts, probs = zip(*outcomes)
        idx = np.random.choice(len(acts), p=np.array(probs))
        return acts[idx], probs[idx]

    def _traverse(self, state, target_player, pi=1.0, pi_op=1.0):
        """
        External‑sampling traversal using state.clone().
        pi    = reach prob of target player
        pi_op = reach prob of opponents
        """
        # Terminal
        if state.is_terminal():
            '''print("Bids: ", state.env.bids)
            all_played = [
                (card.rank, card.suit)
                for trick in state.env.played_tricks
                for card in trick.cards
            ]
            print("Trcks: ", all_played)'''
            return state.returns()[target_player]

        # Chance node
        assert not state.is_chance_node()

        current_player = state.current_player()
        #print(f"Current player: {current_player}")
        #print("current state player: ", state.env._get_current_player())
        obs = state.information_state_tensor()
        legal = state.legal_actions()

        # Compute masked softmax
        if self.policy_player_input:
            logits = self.policy_net(torch.tensor(np.insert(obs, 0, current_player), dtype=torch.float32).unsqueeze(0) ).squeeze(0)
        else:
            logits = self.policy_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0) ).squeeze(0)

        probs = self._masked_softmax(logits, legal)

        # Sample one action
        a_s = int(np.random.choice(legal, p=probs[legal]))
        # Clone and recurse
        next_state = state.clone()
        next_state.apply_action(a_s)
        if current_player == target_player:
            # Counterfactual utility for target
            u_s = self._traverse(next_state, target_player, pi * probs[a_s], pi_op)
            # Baseline from adv network
            v = self.adv_nets[current_player](torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            advantage = (u_s - v[0, a_s].item()) / pi_op
            self.adv_memory[current_player].push(obs, a_s, advantage, current_player)

            with torch.no_grad():
                adv_vals = self.adv_nets[current_player](torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()

            legal_mask = np.isin(np.arange(len(adv_vals)), legal).astype(float)
            adv_val_pos = np.clip(adv_vals, 0, None) * legal_mask
            total_adv_val_pos = adv_val_pos.sum()
            pi_target = adv_val_pos / total_adv_val_pos if total_adv_val_pos > 0 else legal_mask / legal_mask.sum()
            #print("pi target: ", pi_target)
            self.pol_memory.push(obs, pi_target, None, current_player)
                
            return u_s
        else:
            # Opponent node: update opponent reach
            return self._traverse(next_state, target_player, pi, pi_op * probs[a_s])

    def _optimize_adv(self, player):
        if len(self.adv_memory[player]) < self.batch_size:
            return
        batch = self.adv_memory[player].sample(self.batch_size)
        obs = torch.tensor([b.obs for b in batch], dtype=torch.float32)
        acts = torch.tensor([b.action for b in batch], dtype=torch.long)
        advs = torch.tensor([b.advantage for b in batch], dtype=torch.float32)
        preds = self.adv_nets[player](obs)
        chosen = preds.gather(1, acts.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(chosen, advs)
        #print(f"Advantage Loss: {loss.item()}")
        opt = self.adv_opts[player]
        opt.zero_grad(); loss.backward(); opt.step()

    def _optimize_policy(self):
        if len(self.pol_memory) < self.batch_size:
            return
        batch = self.pol_memory.sample(self.batch_size)
        if self.policy_player_input:
            obs_batch = torch.tensor([np.insert(b.obs, 0, b.player) for b in batch], dtype=torch.float32)
        else:
            obs_batch = torch.tensor([b.obs for b in batch], dtype=torch.float32)
        pi_targets = torch.tensor([b.action for b in batch], dtype=torch.float32)

        logits = self.policy_net(obs_batch)
        pred_pi = torch.softmax(logits, dim=1)

        loss = nn.MSELoss()(pred_pi, pi_targets)
        #print(f"Policy Loss: {loss.item()}")
        self.policy_opt.zero_grad(); loss.backward(); self.policy_opt.step()


    def get_policy(self):
        return self.policy_net

    def save(self, prefix):
        for i, net in enumerate(self.adv_nets):
            torch.save(net.state_dict(), f"{prefix}_adv_{i}.pt")
        torch.save(self.policy_net.state_dict(), f"{prefix}_policy.pt")
        print(f"Models saved with prefix '{prefix}'")


    def evaluate_policy(self, policy, num_episodes=50):
        """
        Run full games under the current policy network to estimate
        average return per player over a number of episodes.
        """
        total_returns = np.zeros(self.num_players, dtype=float)
        for _ in range(num_episodes):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                current = state.current_player()
                obs = state.information_state_tensor(current)
                if self.policy_player_input:
                    logits = policy(torch.tensor(np.insert(obs,0,current), dtype=torch.float32).unsqueeze(0)).squeeze(0)
                else:
                    logits = self.policy_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0) ).squeeze(0)


                # mask out illegal actions
                legal = state.legal_actions()
                mask = torch.zeros(self.num_actions, device=logits.device)
                mask[legal] = 1
                unscaled = torch.exp(logits) * mask

                # normalize, detach and bring to CPU numpy
                probs = (unscaled / unscaled.sum())
                probs = probs.detach().cpu().numpy()

                # sample only among legal actions
                #action = np.random.choice(legal, p=probs[legal])
                #take most likely action
                action = legal[np.argmax(probs[legal])]

                #print("legal actions: ", legal)
                #print(f"Action sampled: {action}")
                #print("probs: ", probs.shape)
                #print("legal: ", legal.shape)
                #print("probs legal: ", probs)

                state.apply_action(action)
            
            #print("Bidding history: ", state.env.bids)
            #print("CONTRACT: ", state.env.contract_value)

            total_returns += np.array(state.returns(), dtype=float)
            #print(state.returns())
        return total_returns / num_episodes


class OnlineCFRWrapper:
    def __init__(self, config, env):
        self.config = config
        self.env = env

        self.game = CoincheOpenSpiel()

        self.num_players = self.game.num_players()
        self.num_actions = self.game.num_distinct_actions()


        self.regrets = [np.zeros(self.num_actions) for _ in range(self.num_players)]
        self.strategy_sum = [np.zeros((self.num_actions, )) for _ in range(self.num_players)]

    def learn(self, total_timesteps, callback=None, use_masking=False):
        for t in range(1, total_timesteps + 1):
            state = self.game.new_initial_state()
            self._update_regrets(state)
        
            if t % 100 == 0:
                print(f"Online CFR Iteration {t}/{total_timesteps} done.")

    def _get_strategy(self, player_id, state, legal_actions):
        regrets = np.maximum(self.regrets[player_id], 0)
        sum_regrets = np.sum(regrets[legal_actions])

        strategy = np.zeros(self.num_actions)
        if sum_regrets > 0:
            strategy[legal_actions] = regrets[legal_actions] / sum_regrets
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)

        self.strategy_sum[player_id] += strategy
        return strategy

    def _update_regrets(self, state):
        #print("State : ", state)
        if state.is_terminal():
            return state.returns()

        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)
        #print("Legal actions : ", legal_actions)
        strategy = self._get_strategy(current_player, legal_actions)

        action_utilities = np.zeros(self.num_actions)
        node_utility = 0.0

        for action in legal_actions:
            next_state = state.clone()
            try:
                next_state.apply_action(action)
                utilities = self._update_regrets(next_state)
            except Exception as e:
                print(f"Error applying action {action}: {e}")
                utilities = np.zeros(self.num_players)
                continue
            action_utilities[action] = utilities[current_player]
            node_utility += strategy[action] * utilities[current_player]

        for action in legal_actions:
            regret = action_utilities[action] - node_utility
            self.regrets[current_player][action] += regret

        return [node_utility if player == current_player else 0.0 for player in range(self.num_players)]

    def get_policy(self):
        avg_strategy = []
        for player_strat_sum in self.strategy_sum:
            strat = np.where(player_strat_sum > 0, player_strat_sum, 1)
            strat /= np.sum(strat)
            avg_strategy.append(strat)
        return avg_strategy

    def save(self, filepath):
        #print(f"Saving policy to {filepath}...")
        policy = self.get_policy()
        print("Policy: ", policy)
        policy_dict = {f'player_{i}': policy[i] for i in range(self.num_players)}
        np.savez(filepath, **policy_dict)
        print(f"Policy saved to {filepath}")


