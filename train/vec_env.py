# vec_env.py stays the same
from stable_baselines3.common.vec_env import SubprocVecEnv
from train.wrappers import make_masked_coinche

def make_self_play_vec(num_envs=2, seed=None):
    return SubprocVecEnv(
        [lambda r=i: make_masked_coinche(None if seed is None else seed + r)
         for i in range(num_envs)]
    )
