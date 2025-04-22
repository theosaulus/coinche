from sb3_contrib.ppo_mask import MaskablePPO

def get_algorithm(config, env):
    algo = config["algorithm"].upper()
    common_kwargs = dict(
        policy=config["policy"],
        env=env,
        verbose=1,
        tensorboard_log=f"logs/{algo.lower()}/",
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=config.get("lr", 3e-4),
    )
    if algo == "PPO":
        return MaskablePPO(**common_kwargs)
    else:
        raise NotImplementedError(f"Algorithm {algo} is not implemented.")
