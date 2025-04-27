import numpy as np
import pyspiel
from gym_coinche import GymCoinche



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