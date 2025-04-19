if __name__ == "__main__":
    """Example training loop for Maskable PPO on Coinche."""
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import MaskablePPO
    import torch
    import os

    from train.vec_env import make_self_play_vec

    num_envs = 2
    total_timesteps = 10_000_000
    save_dir = "./checkpoints_ppo_coinche"
    os.makedirs(save_dir, exist_ok=True)

    venv = make_self_play_vec(num_envs=num_envs, seed=42)
    model = MaskablePPO(
        "MultiInputPolicy",  # because observation is Dict with mask
        venv,
        learning_rate=3e-4,
        n_steps=16,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range=0.2,
        verbose=1,
        device="auto",
    )

    chk_callback = CheckpointCallback(
        save_freq=1_000_000 // num_envs, 
        save_path=save_dir,
        name_prefix="ppo_coinche"
    )
    model.learn(total_timesteps=total_timesteps, callback=chk_callback)
    model.save(os.path.join(save_dir, "ppo_coinche_final"))
    print("Training completed – model saved.")
