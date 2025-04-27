import os
import wandb

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback

class GameReplayCallback(BaseCallback):
    def __init__(self, log_every=1000, save_dir='replays/', verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.log_every = log_every
        self.episode_count = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done:
                    self.episode_count += 1
                    if self.episode_count % self.log_every == 0:
                        self.logger.record("game/info", info)
                        self.logger.dump(step=self.num_timesteps)

        return True

def build_callbacks(config):
    callbacks = []

    if config.get("save_game_replays", False):
        callbacks.append(
            GameReplayCallback(
                log_every=config.get("log_every", 1000), 
                save_dir="replays/"
            )
        )

    if config.get("checkpoint_every"):
        callbacks.append(
            CheckpointCallback(
                save_freq=config["checkpoint_every"],
                save_path="models_checkpoints/",
                name_prefix="coinche_model"
            )
        )

    if config.get(wandb, False) and config.get("gradient_save_freq", 0) > 0:
        callbacks.append(WandbCallback(
            gradient_save_freq=config["gradient_save_freq"],
            model_save_freq=config["checkpoint_every"],
            # model_save_path="models_gradients/",  # Path problem on Windows with this
            verbose=1,
        ))

    return callbacks