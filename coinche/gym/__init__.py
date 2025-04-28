from gymnasium.envs.registration import register
from coinche.player import RandomPlayer, AIPlayer, DeterministicPlayer
from coinche.gym.gymplayer import GymPlayer
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium.envs.registration")

register(
    id='coinche-v3',
    entry_point='coinche.gym.env:GymCoinche',
    kwargs={
        'players': [
            RandomPlayer(0, "N"),
            RandomPlayer(1, "E"),
            GymPlayer(2, "S"),
            RandomPlayer(3, "W"),
        ],
    }
)

# register(
#     id='coinche-v4',
#     entry_point='coinche.gym.env:GymCoinche',
#     kwargs={
#         'players': [
#             AIPlayer("./experiments/coinche/11_05_2020-14_54/checkpoint/0_Step-3419.ckpt", 0, "N"),
#             AIPlayer("./experiments/coinche/11_05_2020-14_54/checkpoint/0_Step-3419.ckpt", 1, "E"),
#             GymPlayer(2, "S"),
#             AIPlayer("./experiments/coinche/11_05_2020-14_54/checkpoint/0_Step-3419.ckpt", 3, "W")
#         ],
#         'contrat_model_path': './reward_prediction/reward_model.h5'
#     }
# )


register(
    id='coinche-v4',
    entry_point='coinche.gym.env:GymCoinche',
    kwargs={
        'players': [
            GymPlayer(i, n) for i,n in enumerate(("N","E","S","W")) 
        ],
    }
)

register(
    id='coinche-v5',
    entry_point='coinche.gym.env:GymCoinche',
    kwargs={
        'players': [
            DeterministicPlayer(0, "N"),
            RandomPlayer(1, "E"),
            GymPlayer(2, "S"),
            RandomPlayer(3, "W"),
        ],
    }
)

register(
    id='coinche-v6',
    entry_point='coinche.gym.env:GymCoinche',
    kwargs={
        'players': [
            GymPlayer(0, "N"),
            DeterministicPlayer(1, "E"),
            GymPlayer(2, "S"),
            DeterministicPlayer(3, "W"),
        ],
    }
)

register(
    id='coinche-v7',
    entry_point='coinche.gym.env:GymCoinche',
    kwargs={
        'players': [
            GymPlayer(0, "N"),
            RandomPlayer(1, "E"),
            GymPlayer(2, "S"),
            RandomPlayer(3, "W"),
        ],
    }
)