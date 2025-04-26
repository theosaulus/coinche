import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from coinche.gym.env import GymCoinche

import pyspiel
from open_spiel.python.algorithms import cfr, deep_cfr, nfsp
from open_spiel.python import rl_environment

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
        return np.array(self._observation, dtype=np.float32)

    def information_state_tensor(self, player=None):
        return np.array(self._observation, dtype=np.float32)

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

class OnlineCFRWrapper:
    def __init__(self, config, env):
        self.config = config
        self.env = env

        self.game = CoincheOpenSpiel()

        self.num_players = self.game.num_players()
        self.num_actions = self.game.num_distinct_actions()

        self.regrets = [np.zeros(self.num_actions) for _ in range(self.num_players)]
        self.strategy_sum = [np.zeros(self.num_actions) for _ in range(self.num_players)]

    def learn(self, total_timesteps, callback=None, use_masking=False):
        for t in range(1, total_timesteps + 1):
            state = self.game.new_initial_state()
            self._update_regrets(state)

            #if callback is not None:
            #    callback.on_step(self)

            if t % 100 == 0:
                print(f"Online CFR Iteration {t}/{total_timesteps} done.")

    def _get_strategy(self, player_id, legal_actions):
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


class oldOnlineCFRWrapper:
    def __init__(self, config, env):
        self.config = config
        self.env = env

        self.game = CoincheOpenSpiel()

        self.num_players = self.game.num_players()
        self.num_actions = self.game.num_distinct_actions()

        self.regrets = [np.zeros(self.num_actions) for _ in range(self.num_players)]
        self.strategy_sum = [np.zeros(self.num_actions) for _ in range(self.num_players)]

    def learn(self, total_timesteps, callback=None, use_masking=False):
        for t in range(1, total_timesteps + 1):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                current_player = state.current_player()
                legal_actions = state.legal_actions()

                strategy = self._get_strategy(current_player, legal_actions)
                action = np.random.choice(len(strategy), p=strategy)

                state.apply_action(action)

                # Update regrets after action
                self._update_regrets(current_player)

            #if callback is not None:
            #    callback.on_step(self)

            if t % 100 == 0:
                print(f"Online CFR Iteration {t}/{total_timesteps} done.")

    def _get_strategy(self, player_id, legal_actions):
        regrets = np.maximum(self.regrets[player_id], 0)
        sum_regrets = np.sum(regrets[legal_actions])

        if sum_regrets > 0:
            strategy = np.zeros(self.num_actions)
            strategy[legal_actions] = regrets[legal_actions] / sum_regrets
        else:
            strategy = np.zeros(self.num_actions)
            strategy[legal_actions] = 1.0 / len(legal_actions)

        self.strategy_sum[player_id] += strategy
        return strategy


    def _update_regrets(self, state):
        if state.is_terminal():
            return state.returns() 

        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)
        strategy = self._get_strategy(current_player, legal_actions)

        action_utilities = np.zeros(self.num_actions)
        node_utility = 0.0

        for action in legal_actions:
            next_state = state.clone()
            next_state.apply_action(action)
            utilities = self._update_regrets(next_state)
            action_utilities[action] = utilities[current_player]
            node_utility += strategy[action] * utilities[current_player]

        for action in legal_actions:
            regret = action_utilities[action] - node_utility
            self.regrets[current_player][action] += regret

        return [node_utility if player == current_player else 0.0 for player in range(self.num_players)]
    '''def _update_regrets(self, player_id, legal_actions, action_taken):
        # Simplified regret update (mock reward signals)
        
        reward = self._get_reward(player_id, legal_actions)
        #np.random.uniform(-1, 1)  # Placeholder random reward
        action_rewards = np.full(self.num_actions, reward)

        regret = action_rewards - action_rewards[action_taken]
        self.regrets[player_id] += regret'''

    def get_policy(self):
        avg_strategy = []
        for player_strat_sum in self.strategy_sum:
            strat = np.where(player_strat_sum > 0, player_strat_sum, 1)
            strat /= np.sum(strat)
            avg_strategy.append(strat)
        return avg_strategy

    def save(self, filepath):
        print(f"Saving policy to {filepath}...")
        """Save the learned average strategy to a .npz file."""
        policy = self.get_policy()
        print("Policy: ", policy)
        policy_dict = {f'player_{i}': policy[i] for i in range(self.num_players)}
        np.savez(filepath, **policy_dict)
        print(f"Policy saved to {filepath}")



class DeepCFRWrapper:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.game = CoincheOpenSpiel()
        session = tf.Session()
        self.solver = deep_cfr.DeepCFRSolver(
            session=session,
            game=self.game,
            policy_network_layers=[128,128],#config.policy_network_layers,          # e.g., [64, 64]
            advantage_network_layers=[128,128],#config.value_network_layers,        # e.g., [64, 64]
            num_iterations=100,#config.total_timesteps,                        # e.g., 100
            num_traversals=10,#config.num_traversals_per_iteration,          # e.g., 20
            learning_rate=1e-3,#config.learning_rate,                          # e.g., 1e-4
            batch_size_advantage=64,#config.batch_size,                      # e.g., 128
            batch_size_strategy=64,#config.batch_size,                       # e.g., 128
            memory_capacity=100000,#config.memory_capacity,                      # e.g., 1_000_000
            policy_network_train_steps=1,#config.num_policy_network_epochs, # e.g., 1
            advantage_network_train_steps=1,#config.num_value_network_epochs, # e.g., 1
            reinitialize_advantage_networks=True                         # Typically set to True
)

    def learn(self, total_timesteps, callback=None, use_masking=False):
        self.solver.solve()
        #for it in range(total_timesteps):
        #    self.solver.iteration()
        #    if it % 10 == 0:
         #       print(f"Deep CFR Iteration {it}/{total_timesteps} done.")

    def get_policy(self):
        return self.solver.average_policy()


class CFRWrapper:
    def __init__(self, config, env):
        self.config = config
        self.env = env

        self.game = CoincheOpenSpiel()

        self.num_players = self.game.num_players()
        self.num_actions = self.game.num_distinct_actions()

        self.num_iterations = getattr(config, 'num_iterations', 1000)

        # Initialize the OpenSpiel CFRSolver
        self.solver = cfr.CFRSolver(self.game)

    def learn(self, total_timesteps=None, callback=None, use_masking=False):
        if total_timesteps is None:
            total_timesteps = self.num_iterations

        for t in range(1, total_timesteps + 1):
            self.solver.evaluate_and_update_policy()

            #if callback is not None:
            #    callback.on_step(self)

            if t % 100 == 0:
                print(f"CFR Iteration {t}/{total_timesteps} done.")

    def get_policy(self):
        # Return the average policy computed by the CFRSolver
        average_policy = self.solver.average_policy()
        return average_policy

    def save(self, filepath):
        print(f"Saving policy to {filepath}...")
        """Save the learned average strategy to a .npz file."""
        average_policy = self.get_policy()

        policy_dict = {}
        for state_key in average_policy.policy.keys():
            state_policy = average_policy.policy[state_key]
            policy_dict[state_key] = np.array([state_policy.get(a, 0.0) for a in range(self.num_actions)])

        np.savez(filepath, **policy_dict)
        print(f"Policy saved to {filepath}")



class oldCFRWrapper:
    def __init__(self, config, env):

        self.env = env

        # Initialize your custom Coinche game
        self.game = CoincheOpenSpiel()
        
        # Set the number of iterations from config or default to 1000
        #self.num_iterations = config['num_iterations']
        
        # Initialize the CFR solver
        self.solver = cfr.CFRSolver(self.game)

    def learn(self, total_timesteps, callback=None, use_masking=False):
        # Perform CFR iterations
        for i in range(total_timesteps):
            self.solver.evaluate_and_update_policy()
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1} completed.")

    def get_average_policy(self):
        # Retrieve the average policy learned by CFR
        return self.solver.average_policy()