import os
import wandb
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback

def build_callbacks(config):
    callbacks = []

    if config.get("save_game_replays", False):
        callbacks.append(
            GameReplayCallback(
                log_every=config.get("log_every", 10000), 
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
            model_save_path="models_gradients/",  # Path problem on Windows with this
            verbose=1,
        ))

    if config.get("print_score_stats", True):
        callbacks.append(FinalScoreStatsCallback(
            print_freq=config.get("log_every", 1000),
            log_in_wandb=config.get("wandb", False),
        ))

    return callbacks

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
                        # Log specific keys or a summary of the info dictionary
                        summary_info = {k: v for k, v in info.items() if isinstance(v, (int, float, str))}
                        self.logger.record("game/info", summary_info)
                        self.logger.dump(step=self.num_timesteps)

        return True

class FinalScoreStatsCallback(BaseCallback):
    """
    Records per-episode rewards and key outcome flags for GymPlayers when they act as attacker or defender.
    Periodically prints and optionally logs to Weights & Biases the aggregated statistics.
    """
    def __init__(self, print_freq=1000, log_in_wandb=True, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.log_in_wandb = log_in_wandb
        # Buffers for attacker vs defender
        self.atk_rewards = []
        self.def_rewards = []
        self.atk_success = []
        self.def_success = []
        self.atk_capot_ann = []
        self.def_capot_ann = []
        self.atk_capot_real = []
        self.def_capot_real = []
        self.atk_contr_val = []
        self.def_contr_val = []
        self.atk_coinche_ann = []
        self.def_coinche_ann = []
        self.atk_coinche_real = []
        self.def_coinche_real = []
        self.atk_surcoin_ann = []
        self.def_surcoin_ann = []
        self.atk_surcoin_real = []
        self.def_surcoin_real = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for info, done, rew in zip(infos, dones, rewards):
            if done and "gymplayer_attacker_yn" in info:
                is_atk = bool(info["gymplayer_attacker_yn"])
                is_contract_realized = bool(info.get("contract_realized", False))
                # reward
                (self.atk_rewards if is_atk else self.def_rewards).append(rew)
                # contract realized success
                if is_atk:
                    self.atk_success.append(is_contract_realized)
                else:
                    self.def_success.append(1 - is_contract_realized)
                # capot
                ann = bool(info.get("capot_announced", False))
                real = bool(info.get("capot_realized", False))
                (self.atk_capot_ann if is_atk else self.def_capot_ann).append(ann)
                (self.atk_capot_real if is_atk else self.def_capot_real).append(real)
                # contract value
                val = float(info.get("contract_value", 0))
                (self.atk_contr_val if is_atk else self.def_contr_val).append(val)
                # coinche / surcoinche
                coin = info.get("coinche_surcoinche", 0)
                ann_c = (coin == 1)
                real_c = ann_c and info.get("contract_realized", False)
                (self.atk_coinche_ann if is_atk else self.def_coinche_ann).append(ann_c)
                (self.atk_coinche_real if is_atk else self.def_coinche_real).append(real_c)
                # surcoinche
                ann_s = (coin == 2)
                real_s = ann_s and info.get("contract_realized", False)
                (self.atk_surcoin_ann if is_atk else self.def_surcoin_ann).append(ann_s)
                (self.atk_surcoin_real if is_atk else self.def_surcoin_real).append(real_s)

        # print & optionally log
        if self.n_calls % self.print_freq == 0 and (self.atk_rewards or self.def_rewards):
            print(f"\n=== Performance over {len(self.atk_rewards)+len(self.def_rewards)} episodes ===")
            metrics = {}
            def summarize(prefix, buf):
                if buf:
                    mean = np.mean(buf)
                    count = len(buf)
                    print(f" {prefix}: {mean:.3f} ({count})")
                    metrics[f"{prefix}"] = float(mean)

            # rewards
            summarize("As attacker reward_mean", self.atk_rewards)
            summarize("As defender reward_mean", self.def_rewards)
            # success
            summarize("As attacker success_rate", self.atk_success)
            summarize("As defender success_rate", self.def_success)
            # capot
            summarize("Capot announce rate", self.atk_capot_ann)
            summarize("Capot realisation rate", self.atk_capot_real)
            summarize("Opponent capot announce rate", self.def_capot_ann)
            summarize("Opponent capot realisation rate", self.def_capot_real)
            # contract value
            summarize("Contract value mean", self.atk_contr_val)
            summarize("Opponent contract value mean", self.def_contr_val)
            # coinche
            summarize("Coinche announce rate", self.atk_coinche_ann)
            summarize("Coinche win rate", self.atk_coinche_real)
            summarize("Opponent coinche announce rate", self.def_coinche_ann)
            summarize("Opponent coinche win rate", self.def_coinche_real)
            # surcoinche
            summarize("Surcoinche announce rate", self.atk_surcoin_ann)
            summarize("Surcoinche win rate", self.atk_surcoin_real)
            summarize("Opponent surcoinche announce rate", self.def_surcoin_ann)
            summarize("Opponent surcoinche win rate", self.def_surcoin_real)
            print()

            if self.log_in_wandb:
                wandb.define_metric("*")
                wandb.log(metrics)

            # clear all
            for buf in [self.atk_rewards, self.def_rewards,
                        self.atk_success, self.def_success,
                        self.atk_capot_ann, self.def_capot_ann,
                        self.atk_capot_real, self.def_capot_real,
                        self.atk_contr_val, self.def_contr_val,
                        self.atk_coinche_ann, self.def_coinche_ann,
                        self.atk_coinche_real, self.def_coinche_real,
                        self.atk_surcoin_ann, self.def_surcoin_ann,
                        self.atk_surcoin_real, self.def_surcoin_real]:
                buf.clear()

        return True
