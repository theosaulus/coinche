import os
import wandb

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback

class GameReplayCallback(BaseCallback):
    def __init__(self, save_dir='replays/', verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Custom logic for saving state/action history goes here
        # Save to file every N episodes (optional)
        return True

def build_callbacks(config):
    callbacks = []

    if config.get("save_game_replays", False):
        callbacks.append(GameReplayCallback())

    if config.get("checkpoint_every"):
        callbacks.append(
            CheckpointCallback(
                save_freq=config["checkpoint_every"],
                save_path="models/",
                name_prefix="coinche_model"
            )
        )

    callbacks.append(WandbCallback(
        gradient_save_freq=100,
        # model_save_path="models/", #This caused an error with wandb...
        verbose=1,
    ))

    return callbacks