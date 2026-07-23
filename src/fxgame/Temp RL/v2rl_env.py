"""
================================================================================
 v2rl_env.py  --  a continuous, grid-free RL environment for the v2 game
================================================================================

 Same GAME as v2_sequential_offers.py:
   * four trading rounds; up to K sequential sized (P, q) offers per round;
   * a Bureau de Change (BdC) dump each round once the rate is revealed;
   * settlement at the round-5 rate with residue and deficit penalties.

 Different SOLVER: nothing here is on a grid. Prices, sell sizes and BdC dumps
 are real-valued, and the settlement rate a5 is sampled directly -- there is no
 kappa (v2's kappa only existed to keep its value function deterministic). The
 MM accept rule and the BdC fee are imported from fx_mechanics, so this env and
 v2 provably play the same game; the only differences are the removed grids.

 Why this exists: to check that a model-free agent RECOVERS v2's exact-up-to-
 grid answer (~ -0.36% P/L on the T1 card) before we trust RL in v3, where no
 exact answer exists.

 --- The MDP ------------------------------------------------------------------
 One episode = one whole game. Each round is K offer-steps (rate X hidden) then
 one BdC-step (rate X revealed), so an episode is rounds*(K+1) steps, and the
 reward is zero until a single terminal payoff = P/L. Within a round the hidden
 rate X ~ N(a, sd^2) is drawn ONCE and reused across the K offers, exactly as
 the MM has one true rate per round.

 Observation (8 floats), all continuous, no grids:
     0  rounds_left / rounds          (time-left across rounds)
     1  offers_left / K               (offers left in this round; 0 at BdC)
     2  phase                         (0 = offer step, 1 = BdC step)
     3  c / L                         (pounds remaining, fraction)
     4  d / T                         (dollars banked, fraction of target)
     5  (a - a0) / sd                 (anchor = last revealed rate, in sds)
     6  lo_z                          (belief floor on X,  z vs anchor)
     7  hi_z                          (belief ceiling on X, z vs anchor)
 The belief bracket (6,7) is the sufficient statistic of the within-round offer
 history -- exposing it makes the state Markov and gives the agent exactly the
 information v2's DP state carries (fairest possible comparison). An open floor
 / ceiling is shown as -+Z_CLAMP; at the BdC step the bracket collapses to the
 revealed rate.

 Action (2 floats in [0, 1]):
     offer step:  a[0] -> price z in [-Z_OFFER, +Z_OFFER], P = a + sd*z
                  a[1] -> sell fraction of current pounds, q = a[1] * c
                          (q = 0 is a pure probe: no pounds move, bracket still
                           tightens -- and "stop and go to BdC" is just probing
                           out the rest of the round, so no explicit stop action)
     BdC step:    a[1] -> dump fraction of current pounds, m = a[1] * c
                  a[1] -> ignored
================================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field

from fx_mechanics import GameParams, mm_accepts, bdc_payoff_per_pound

Z_CLAMP = 5.0     # an open belief bracket is reported as +-Z_CLAMP (z-units)
Z_OFFER = 4.0     # offer price spans anchor +- Z_OFFER sds (matches v2's grid span)


@dataclass
class GameSpec:
    """The trader. Defaults are the T1 card -- identical numbers to v2's TraderSpec."""
    L: float = 100_000.0          # starting pounds
    T: float = 125_000.0          # dollar target
    A: float = 0.02               # residue penalty on leftover pounds
    B: float = 0.03               # deficit penalty on the dollar shortfall below T
    rounds: int = 4               # trading rounds (round 5 = settlement draw only)
    K: int = 3                    # sequential offers per round
    params: GameParams = field(default_factory=GameParams)


class V2Game(gym.Env):
    """The v2 game as a Gymnasium environment (continuous, no grids)."""

    def __init__(self, spec: GameSpec = None):
        super().__init__()
        self.spec = spec or GameSpec()
        p = self.spec.params
        self.a0, self.sd, self.f = p.a0, p.sd, p.bdc_fee

        obs_low = np.array([0, 0, 0, 0, -np.inf, -np.inf, -Z_CLAMP, -Z_CLAMP],
                           dtype=np.float32)
        obs_high = np.array([1, 1, 1, 1, np.inf, np.inf, Z_CLAMP, Z_CLAMP],
                            dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)#cont vector with upper lower bounds
        # action = [z, fraction]: z = offer price in sds from the CURRENT anchor
        # (P = a + sd*z); fraction = share of remaining pounds to sell / dump.
        act_low = np.array([-Z_OFFER, 0.0], dtype=np.float32)
        act_high = np.array([Z_OFFER, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    # ---- settlement (identical accounting to v2's terminal_wealth) -----------
    def _settle(self, c, d, a5):
        s = self.spec
        deficit = max(0.0, s.T - d)
        return c * (1.0 - s.A) + (d - s.B * deficit) / a5

    # ---- observation ---------------------------------------------------------
    def _obs(self):
        s = self.spec
        rounds_left = (s.rounds - self.n + 1) / s.rounds #normalises as [0,1]
        offers_left = self.k / s.K
        a_z = (self.a - self.a0) / self.sd
        if self.phase == 0:                      # offers: X hidden -> show bracket
            lo_z = (-Z_CLAMP if self.lo == -np.inf
                    else float(np.clip((self.lo - self.a) / self.sd, -Z_CLAMP, Z_CLAMP)))
            hi_z = (Z_CLAMP if self.hi == np.inf
                    else float(np.clip((self.hi - self.a) / self.sd, -Z_CLAMP, Z_CLAMP)))
        else:                                    # BdC: X revealed -> bracket = X
            xz = float(np.clip((self.X - self.a) / self.sd, -Z_CLAMP, Z_CLAMP))
            lo_z = hi_z = xz
        return np.array([rounds_left, offers_left, float(self.phase),
                         self.c / s.L, self.d / s.T, a_z, lo_z, hi_z],
                        dtype=np.float32)

    # ---- round setup ---------------------------------------------------------
    def _start_round(self):
        self.k = self.spec.K
        self.phase = 0
        self.lo, self.hi = -np.inf, np.inf
        # the hidden true rate for this round, drawn ONCE (MM has one rate/round).
        # A forced path (validation only) overrides the draw.
        if self._force_path is not None:
            self.X = float(self._force_path[self.n - 1])
        else:
            self.X = self.a + self.sd * self.rng.standard_normal()
    #Starts a new episode
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        # optional forced rates for path-by-path validation:
        #   options={"path": [X_1..X_rounds], "a5": <settlement rate>}
        self._force_path = options.get("path") if options else None
        self._force_a5 = options.get("a5") if options else None
        self.n = 1
        self.a = self.a0
        self.c, self.d = self.spec.L, 0.0
        self._start_round()
        return self._obs(), {}

    # ---- one step ------------------------------------------------------------
    def step(self, action):
        s = self.spec
        reward, terminated = 0.0, False

        if self.phase == 0:
            # --- an MM offer at P = anchor + sd*z ---
            z = float(np.clip(action[0], -Z_OFFER, Z_OFFER))   # price, sds from anchor
            q = float(np.clip(action[1], 0.0, 1.0)) * self.c   # fraction of pounds
            P = self.a + self.sd * z
            if mm_accepts(P, self.X):            # P < X
                self.c -= q
                self.d += P * q
                self.lo = max(self.lo, P)        # learned X > P (floor)
            else:
                self.hi = min(self.hi, P)        # learned X <= P (ceiling)
            self.k -= 1
            if self.k == 0:
                self.phase = 1                   # next: BdC (reveals X)
        else:
            # --- the BdC dump at the now-revealed rate X ---
            # at the BdC only a fraction matters; it is carried in action[1]
            # (slot 1 is always the fraction; slot 0, the price, is unused here).
            m = float(np.clip(action[1], 0.0, 1.0)) * self.c
            self.d += bdc_payoff_per_pound(self.X, self.f) * m
            self.c -= m
            self.a = self.X                      # revealed rate becomes the anchor
            self.n += 1
            if self.n > s.rounds:                # settlement
                a5 = (float(self._force_a5) if self._force_a5 is not None
                      else self.a + self.sd * self.rng.standard_normal())
                W = self._settle(self.c, self.d, a5)
                reward = (W - s.L) / s.L         # terminal P/L (fraction)
                terminated = True
            else:
                self._start_round()

        if terminated:
            obs = np.array([0, 0, 1, self.c / s.L, self.d / s.T,
                            (self.a - self.a0) / self.sd, 0.0, 0.0], dtype=np.float32)
        else:
            obs = self._obs()
        return obs, reward, terminated, False, {}


class GridV2Game(V2Game):
    """DQN plays the game on v2's OWN action grid, for a like-for-like compare.

    The continuous [price_lever, fraction] box is replaced by v2's action set:
    the Cartesian product of a 49-point z-price grid and v2's quantity-fraction
    grid (its f_all), plus one stop action. Mechanics, observation, reward and
    the forced-path hook are all inherited unchanged from V2Game -- only the
    action interface differs, so DQN here optimises over exactly the grid the
    v2 DP does. 'stop' offers nothing (a probe), which is outcome-equivalent to
    v2 stopping and heading to the BdC -- the same emulation the DP-replay gate
    is built on.
    """

    def __init__(self, spec: GameSpec = None, n_prices: int = 49, n_quantity_base: int = 24):
        super().__init__(spec or GameSpec())
        self.price_z = np.linspace(-Z_OFFER, Z_OFFER, n_prices)          # v2's z grid
        self.quantity_grid = np.unique(                                  # v2's f_all
            np.concatenate(([0.002, 0.01], np.linspace(0.0, 1.0, n_quantity_base))))
        self.n_quantity = len(self.quantity_grid)
        self.stop_action = n_prices * self.n_quantity
        self.action_space = spaces.Discrete(self.stop_action + 1)

    def decode_action(self, action):
        """Discrete index -> the underlying [z, fraction] the parent step wants.
        At an offer, action[0] is the price in sds from anchor and action[1]
        the sell fraction; at the BdC the dump fraction is in action[1]."""
        action = int(action)
        if self.phase == 1:                      # BdC: only the dump fraction matters
            frac = 0.0 if action == self.stop_action else self.quantity_grid[action % self.n_quantity]
            return np.array([0.0, frac], np.float32)
        if action == self.stop_action:           # stop = offer nothing (probe at z=0)
            return np.array([0.0, 0.0], np.float32)
        price_i, qty_i = divmod(action, self.n_quantity)
        return np.array([self.price_z[price_i], self.quantity_grid[qty_i]], np.float32)

    def step(self, action):
        return super().step(self.decode_action(action))