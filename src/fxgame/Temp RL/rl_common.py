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
    
class PotentialShaping(gym.Wrapper):
    """Dense, optimum-preserving reward shaping (Ng, Harada & Russell 1999).
 
    Adds F = Phi(s') - Phi(s) to every step's reward, where Phi is the
    mark-to-market P/L of the current book:
 
        Phi(s) = [ c(1-A) + (d - B*max(T-d, 0)) / a ] / L  -  1
                 (shifted so Phi(start) = 0; Phi(terminal) := 0)
 
    Consequences, both exact:
      * the per-EPISODE shaped return equals the true P/L -- the constants
        telescope away -- so learning curves stay comparable to the sparse
        runs and to the always-BdC / DP reference lines;
      * the optimal policy is provably unchanged, because every policy's
        return shifts by the same telescoping constant (zero here).
 
    What changes is only WHEN reward arrives: selling above the mark pays
    immediately, carrying pounds toward a penalty costs immediately. The mark
    uses 1/a rather than kappa = E[1/a5]; any potential is valid, this one is
    just cheap. Wrap the TRAINING env only -- diagnostics read the raw env.
    """
 
    def __init__(self, env):
        super().__init__(env)
        self._phi0 = 0.0
        self._prev = 0.0
 
    def _phi_raw(self):
        u = self.env.unwrapped
        s = u.spec
        mtm = u.c * (1.0 - s.A) + (u.d - s.B * max(s.T - u.d, 0.0)) / u.a
        return mtm / s.L - 1.0
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi0 = self._phi_raw()
        self._prev = 0.0                       # Phi~(s0) = 0 by the shift
        return obs, info
 
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        cur = 0.0 if (term or trunc) else self._phi_raw() - self._phi0
        shaped = r + cur - self._prev
        self._prev = cur
        return obs, shaped, term, trunc, info
 
 
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
 
 
class DoubleDQN(DQN):
    """DQN with the Double-DQN target (van Hasselt et al. 2016).
 
    Vanilla DQN bootstraps from max_a Q_target(s', a): the SAME network both
    picks and scores the next action, so estimation noise is always harvested
    at its most optimistic -- the overestimation bias diagnosed in the v2 RL
    review (section 4.2). Double-DQN decouples the two roles:
 
        a* = argmax_a Q_online(s', a)          (online net SELECTS)
        y  = r + (1 - done) * g * Q_target(s', a*)   (target net SCORES)
 
    Everything else -- replay, epsilon schedule, Huber loss, target sync -- is
    inherited from SB3's DQN unchanged. train() below is SB3 2.9.0's own
    train() with the target computation factored into _compute_target and the
    two lines above swapped in.
    """
 
    def _compute_target(self, replay_data, discounts):
        with th.no_grad():
            # online net SELECTS the next action ...
            next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
            # ... target net SCORES it
            next_q_values = th.gather(
                self.q_net_target(replay_data.next_observations), dim=1, index=next_actions)
            return replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
 
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
 
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
 
            target_q_values = self._compute_target(replay_data, discounts)
 
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
 
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
 
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
 
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
