if __name__ == "__main__": # This precaution is just for Windows users
    import torch
    import torch.nn as nn
    import numpy as np
    import gymnasium as gym
    from gymnasium.vector import AsyncVectorEnv

    # Minimal torch policy
    class TorchPolicy(nn.Module):
        def __init__(self, input_dim=98, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.net(x)

    # Factory to create parallel envs
    def make_env():
        def _init():
            import coinche.gym
            return gym.make("coinche-v3")
        return _init

    # Create environments
    envs = AsyncVectorEnv([make_env() for _ in range(4)])
    obs, _ = envs.reset()

    # === Test with Random Policy ===
    done = [False] * 4
    while not all(done):
        random_actions = np.random.rand(4, 32).astype(np.float32)
        obs, rewards, terminated, truncated, infos = envs.step(random_actions)
        done = [t or tr for t, tr in zip(terminated, truncated)]

    print("✅ Random test completed. Final rewards:", rewards)

    # === Test with Trainable Torch Policy ===
    model = TorchPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32)

    for step in range(5):  # Very short training loop
        logits = model(obs)
        actions = logits.detach().numpy()

        obs_np, rewards, terminated, truncated, infos = envs.step(actions)
        obs = torch.tensor(obs_np, dtype=torch.float32)

        # Dummy reward-based loss
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        loss = -torch.mean(logits * reward_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step+1} | Avg reward: {np.mean(rewards):.3f}")

    print("✅ Torch test completed.")
