from coinche.gym.env import make_env
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

def make_env_with_masking(env_id):
    def _wrapped():
        env = make_env(env_id=env_id)()
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _wrapped


def mask_fn(env):
    # Not used directly yet: stub for when GymPlayer supports invalid action masking
    if hasattr(env, 'get_action_mask'):
        return env.get_action_mask()
    else:
        return [1] * env.action_space.n  # All actions are valid
