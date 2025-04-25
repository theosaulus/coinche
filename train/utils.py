from coinche.gym.env import make_env
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

def make_env_with_masking(env_id):
    def _wrapped():
        env = make_env(env_id=env_id)()
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _wrapped


def mask_fn(env):
    # env = getattr(env, "env", env) # Unwrap the Monitor if necessary
    # env = getattr(env, "env", env) # Unwrap the ActionMasker if necessary
    # if hasattr(env, 'get_action_mask'):
    #     return env.get_action_mask()
    # unwrap all wrappers to get to the base environment, and thus the action mask
    real = env
    while hasattr(real, "env"):
        real = real.env
    
    if hasattr(real, 'get_action_mask'):
        return real.get_action_mask()
    else:
        print("Warning: No action mask function found. Returning all actions as valid.")
        return [1] * env.action_space.n  # All actions are valid
