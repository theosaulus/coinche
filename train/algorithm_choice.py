from sb3_contrib.ppo_mask import MaskablePPO
import torch

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


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
    else:
        raise NotImplementedError(f"Algorithm '{algo}' is not supported yet.")