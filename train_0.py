import os
import yaml
import wandb
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv

from train.callbacks import build_callbacks
from train.utils import make_env_with_masking
from train.algorithm_choice import get_algorithm


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    if config["wandb"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True
        )

    env_fn = make_env_with_masking(config['env_id'])
    env = DummyVecEnv([env_fn])

    algo = config["algorithm"].upper()
    model = get_algorithm(config, env)
    callbacks = build_callbacks(config)
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks
    )

    model.save(os.path.join("models", f"{algo.lower()}_final"))
    if config["wandb"]:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train/config.yaml')
    args = parser.parse_args()
    main(args.config)