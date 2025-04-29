import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sb3_contrib.ppo_mask import MaskablePPO
import torch

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

from sb3_contrib import QRDQN

from train.wrapper import DeepCFRWrapper, OnlineCFRWrapper
#from coinche.gym.env import GymCoinche

#import pyspiel
#from open_spiel.python.algorithms import cfr, deep_cfr, nfsp
#from open_spiel.python import rl_environment

def get_algorithm(config, env):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    algo = config["algorithm"].upper()
    if algo == "PPO":
        ppo = MaskablePPO(
            policy=config.get("policy"),
            env=env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            clip_range_vf=config.get("clip_range_vf", None),
            normalize_advantage=config.get("normalize_advantage", True),
            ent_coef=config.get("ent_coef", 0.0),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            target_kl=config.get("target_kl", None),
            tensorboard_log="logs/ppo/",
            verbose=1,
            seed=config.get("seed", None),
            device=device,
        )
        # if hasattr(torch, "compile"):
        #     ppo.policy = torch.compile(
        #         ppo.policy,
        #         fullgraph=False,        # try to fuse as much as possible
        #         dynamic=True           # keep control‐flow dynamic if needed
        #     )
        return ppo
    elif algo == "DCFR": 
        return DeepCFRWrapper(config)

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
    #elif algo == "CFR": 
    #    return CFRWrapper(config, env)
    else:
        raise NotImplementedError(f"Algorithm '{algo}' is not supported yet.")