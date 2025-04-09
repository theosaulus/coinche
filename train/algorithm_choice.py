from sb3_contrib.ppo_mask import MaskablePPO

def get_algorithm(config, env):
    algo = config["algorithm"].upper()
    if algo == "PPO":
        return MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="logs/ppo/",
        )
    else:
        raise NotImplementedError(f"Algorithm '{algo}' is not supported yet.")
