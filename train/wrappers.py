# wrappers.py  (replace the helper and wrapper construction)

import numpy as np
import gymnasium as gym

from coinche.gym.gymplayer import GymPlayer
from sb3_contrib.common.wrappers import ActionMasker
from coinche.utils import convert_index_to_cards
from coinche.card import Suit


# ------------------------------------------------------------------
# Boolean‑mask helper
# ------------------------------------------------------------------
def action_mask(env: gym.Env) -> np.ndarray:
    """
    Return a bool array of shape (n_actions,) where True marks a legal
    action for the *current* player.  Always works on env.unwrapped so
    we bypass OrderEnforcing and other wrappers.
    """
    base = env.unwrapped                       # <- GymCoinche
    n     = base.action_space.n
    mask  = np.zeros(n, dtype=np.bool_)

    # -------- bidding phase --------
    if not base.bidding_done:
        mask[base._get_valid_bid_actions(base.current_bid)] = True
        return mask

    # -------- trick phase ----------
    trick        = base.trick
    suits_order  = base.suits_order
    player       = base.current_trick_rotation[0]

    # cards 0‑31
    for idx in range(32):
        card = convert_index_to_cards([idx], suits_order)[0]
        if not player.has_card(card):
            continue
        if trick.highest_card is None:      # <- works regardless of list/len
            mask[idx] = True
            continue
        mask[idx] = trick._assert_valid_play_TrueFalse(card, player)

    # coinche / surcoinche / pass
    if base.coinche_surcoinche == 0 and base.current_bid is not None:
        mask[41] = True       # coinche
    elif base.coinche_surcoinche == 1:
        mask[42] = True       # sur‑coinche
    mask[43] = True           # pass always legal
    return mask


# ------------------------------------------------------------------
# Optional observation wrapper that injects the mask
# ------------------------------------------------------------------
class ObsMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        m_space = gym.spaces.Box(0, 1, (env.action_space.n,), dtype=np.bool_)
        self.observation_space = gym.spaces.Dict(
            {**env.observation_space.spaces, "action_mask": m_space}
        )

    def _aug(self, obs):
        obs = obs.copy()
        obs["action_mask"] = action_mask(self)
        return obs

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        return self._aug(o), info

    def step(self, a):
        o, r, t, tr, info = self.env.step(a)
        return self._aug(o), r, t, tr, info


class TurnSkippingWrapper(gym.Wrapper):
    """
    Single‑agent view: the wrapped env only asks the outer RL policy
    when the internal GymPlayer seat must act.  After the policy returns
    an action, we auto‑advance the environment (letting other seats play
    with their built‑in policies) until GymPlayer is on turn again or
    the round ends.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # keep a reference to the unique GymPlayer seat
        self._gym_seat = next(
            p for p in env.unwrapped.players if isinstance(p, GymPlayer)
        )

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _advance_until_gymplayer(self):
        """
        Mutate the underlying env until GymPlayer must act or the episode
        ends.  Returns cum_reward and a done flag.
        """
        cum_r = 0.0
        base  = self.env.unwrapped

        while True:
            # ----- terminal check ----------------------------------------
            if base.bidding_done and len(base.played_tricks) == 8:
                # round ended during auto‑play
                return cum_r, True

            # ----- is it our turn yet ? ----------------------------------
            if (not base.bidding_done and
                    base.players[base.current_bidding_player_index] is self._gym_seat):
                return cum_r, False
            if (base.bidding_done and
                    base.current_trick_rotation[0] is self._gym_seat):
                return cum_r, False

            # ----- let the built‑in logic play one move ------------------
            if not base.bidding_done:
                # automatic bidder
                p       = base.players[base.current_bidding_player_index]
                vb      = base._get_valid_bid_actions(base.current_bid)
                obs     = base._get_current_observation()
                a       = p.bid(obs, vb, base.suits_order or list(Suit))
                base._process_bidding(a, p)
                base.current_bidding_player_index = (base.current_bidding_player_index + 1) % 4
            else:
                # automatic card play
                p = base.current_trick_rotation[0]
                obs = base._get_current_observation()
                p.play_trick(base.trick, obs, base.suits_order)
                base.current_trick_rotation.pop(0)
                base._play_until_end_of_rotation_or_ai_play()

                # trick reward bookkeeping
                if base.trick.is_done():
                    winner = base.trick.winner
                    tf = (self._gym_seat.index % 2 == winner.index % 2) * base.tricks_reward_factor
                    cum_r += base._get_trick_reward(base.trick, tf)
                    base.played_tricks.append(base.trick)

            # loop until it's GymPlayer's turn or episode done

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        extra, done = self._advance_until_gymplayer()
        if done:
            return self.reset(**kw)            # round ended before our turn
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        extra, done = self._advance_until_gymplayer()
        r += extra
        if done:
            term = True
        return obs, r, term, trunc, info


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def make_masked_coinche(seed=None):
    from coinche.gym.env import make_env
    env = make_env(seed=seed)()
    env = TurnSkippingWrapper(env)            
    env = ActionMasker(env, action_mask)   # <- SB3‑contrib wrapper
    env = ObsMaskWrapper(env)              # (optional)
    return env
