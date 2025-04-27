import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib import QRDQN

from train.wrapper import CFRWrapper, DeepCFRWrapper, OnlineCFRWrapper
#from coinche.gym.env import GymCoinche

#import pyspiel
#from open_spiel.python.algorithms import cfr, deep_cfr, nfsp
#from open_spiel.python import rl_environment




def get_algorithm(config, env):
    algo = config["algorithm"].upper()
    if algo == "PPO":
        return MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="logs/ppo/",
        )
    elif algo == "CFR": 
        return CFRWrapper(config, env)
    elif algo == "DCFR": 
        return DeepCFRWrapper(config, env)

    elif algo == "OCFR": #DO Deep CFR
        #game = pyspiel.load_game("coinche")
        return OnlineCFRWrapper(config, env)
    
    elif algo == "QRDQN":
        #need to fix if we want to implement -- model.learn( --> TypeError: QRDQN.learn() got an unexpected keyword argument 'use_masking'
        policy_kwargs = dict(n_quantiles=50)
        return QRDQN("MlpPolicy", 
                     env, 
                     policy_kwargs=policy_kwargs, 
                     verbose=1,
                     tensorboard_log="logs/ppo/")
    
    else:
        raise NotImplementedError(f"Algorithm '{algo}' is not supported yet.")