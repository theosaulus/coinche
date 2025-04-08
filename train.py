import os
import yaml
import wandb
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from train.callbacks import build_callbacks
from train.utils import make_env_with_masking


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
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
    if algo == "PPO":
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="logs/ppo/",
        )
    else:
        raise NotImplementedError(f"Algorithm '{algo}' is not supported yet.")

    callbacks = build_callbacks(config)

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks
    )

    model.save(os.path.join("models", f"{algo.lower()}_final"))
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train/config.yaml')
    args = parser.parse_args()
    main(args.config)