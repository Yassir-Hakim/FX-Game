"""Training helper shared by train_ppo.py and train_dqn.py."""

import gymnasium as gym


class RewardScale(gym.RewardWrapper):
    """Scale the terminal P/L (~1e-2) up to '% P/L' (~1e0). A constant scale
    does not change the optimal policy; it just hands the agent a comfortable
    reward/value magnitude.

    The env itself keeps reporting the game's TRUE P/L fraction -- the
    diagnostics and the DP-replay gate read that directly -- so the x100 is
    applied here, at the training boundary, and nowhere else."""

    def __init__(self, env, scale=100.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale