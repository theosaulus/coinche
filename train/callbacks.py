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
    def __init__(self, print_freq = 1000, log_in_wandb = True, verbose = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.log_in_wandb = log_in_wandb
        self._infos: list = []

    def _on_step(self) -> bool:
        # In VecEnv contexts, `infos` is a list of dicts, one per env
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "total_attacker_points" in info:
                self._infos.append(info)

        # Periodically print & reset
        if self.n_calls % self.print_freq == 0 and self._infos:
            self._log_to_wandb(self.log_in_wandb)
            self._infos.clear()
        return True
    
    def _log_to_wandb(self, log_in_wandb):
        arr = self._infos
        gym_atk_yn = np.array([i["gymplayer_attacker_yn"] for i in arr])
        atk_pts   = np.array([i["total_attacker_points"] for i in arr])
        def_pts   = np.array([i["total_defender_points"] for i in arr])
        succ      = np.array([i["contract_realized"] for i in arr], dtype=float)
        cap_ann   = np.array([i["capot_announced"] for i in arr], dtype=float)
        cap_succ  = np.array([i["capot_realized"] for i in arr], dtype=float)
        belote    = np.array([i["belote"] for i in arr], dtype=float)
        contr_val = np.array([i["contract_value"] for i in arr], dtype=float)
        coinche   = np.array([i.get("coinche_surcoinche", 0)==1 for i in arr], dtype=float)
        surcoin   = np.array([i.get("coinche_surcoinche", 0)==2 for i in arr], dtype=float)

        atk_pts_gym_true   = atk_pts[gym_atk_yn]
        def_pts_gym_true   = def_pts[gym_atk_yn]
        succ_gym_true      = succ[gym_atk_yn]
        cap_ann_gym_true   = cap_ann[gym_atk_yn]
        cap_succ_gym_true  = cap_succ[gym_atk_yn]
        contr_val_gym_true = contr_val[gym_atk_yn]
        coinche_gym_true   = coinche[gym_atk_yn]
        surcoin_gym_true   = surcoin[gym_atk_yn]

        atk_pts_gym_false   = atk_pts[~gym_atk_yn]
        def_pts_gym_false   = def_pts[~gym_atk_yn]
        succ_gym_false      = succ[~gym_atk_yn]
        cap_ann_gym_false   = cap_ann[~gym_atk_yn]
        cap_succ_gym_false  = cap_succ[~gym_atk_yn]
        contr_val_gym_false = contr_val[~gym_atk_yn]
        coinche_gym_false   = coinche[~gym_atk_yn]
        surcoin_gym_false   = surcoin[~gym_atk_yn]
        
        print(f"\n=== Final‐Score Stats over {len(arr)} episodes ===")
        # print(f"Overall:")
        # print(f" Attacker pts: {atk_pts.mean():.1f}  Defender pts: {def_pts.mean():.1f}")
        # print(f" Success rate       : {succ.mean()*100:.1f}%")
        # print(f" Contract value avg : {contr_val.mean():.1f}")
        # print(f" Capot ann/succ     : {cap_ann.mean()*100:.1f}% / {cap_succ.mean()*100:.1f}%")
        # print(f" Coinche / Surcoin.: {coinche.mean()*100:.1f}% / {surcoin.mean()*100:.1f}%\n")

        print(f"Gym player attacker (True):")
        print(f" Attacker pts: {atk_pts_gym_true.mean():.1f}  Defender pts: {def_pts_gym_true.mean():.1f}")
        print(f" Success rate       : {succ_gym_true.mean()*100:.1f}%")
        print(f" Contract value avg : {contr_val_gym_true.mean():.1f}")
        print(f" Capot ann/succ     : {cap_ann_gym_true.mean()*100:.1f}% / {cap_succ_gym_true.mean()*100:.1f}%")
        print(f" Coinche / Surcoin.: {coinche_gym_true.mean()*100:.1f}% / {surcoin_gym_true.mean()*100:.1f}%\n")

        print(f"Gym player defender:")
        print(f" Attacker pts: {atk_pts_gym_false.mean():.1f}  Defender pts: {def_pts_gym_false.mean():.1f}")
        print(f" Success rate       : {succ_gym_false.mean()*100:.1f}%")
        print(f" Contract value avg : {contr_val_gym_false.mean():.1f}")
        print(f" Capot ann/succ     : {cap_ann_gym_false.mean()*100:.1f}% / {cap_succ_gym_false.mean()*100:.1f}%")
        print(f" Coinche / Surcoin.: {coinche_gym_false.mean()*100:.1f}% / {surcoin_gym_false.mean()*100:.1f}%\n")

        metrics = {
            "final/attacker_pts_mean_gym_true":      atk_pts_gym_true.mean(),
            "final/defender_pts_mean_gym_true":      def_pts_gym_true.mean(),
            "final/contract_success_rate_gym_true":  succ_gym_true.mean(),
            "final/contract_value_mean_gym_true":    contr_val_gym_true.mean(),
            "final/capot_announced_rate_gym_true":   cap_ann_gym_true.mean(),
            "final/capot_success_rate_gym_true":     cap_succ_gym_true.mean(),
            "final/coinche_rate_gym_true":           coinche_gym_true.mean(),
            "final/surcoinche_rate_gym_true":        surcoin_gym_true.mean(),
            "final/attacker_pts_mean_gym_false":     atk_pts_gym_false.mean(),
            "final/defender_pts_mean_gym_false":     def_pts_gym_false.mean(),
            "final/contract_success_rate_gym_false": succ_gym_false.mean(),
            "final/contract_value_mean_gym_false":   contr_val_gym_false.mean(),
            "final/capot_announced_rate_gym_false":  cap_ann_gym_false.mean(),
            "final/capot_success_rate_gym_false":    cap_succ_gym_false.mean(),
            "final/coinche_rate_gym_false":          coinche_gym_false.mean(),
            "final/surcoinche_rate_gym_false":       surcoin_gym_false.mean(),
        }
        # use the number of timesteps as the step for plotting
        if log_in_wandb:
            wandb.define_metric("final/*", step_metric="global_step")
            metrics["global_step"] = self.num_timesteps
            wandb.log(metrics)
            if self.verbose:
                print(f"[W&B] logged final‐score metrics at step {self.num_timesteps}")
