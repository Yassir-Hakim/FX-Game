import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
================================================================================
 rl_env.py  --  the trading game as a Gymnasium environment
================================================================================

 n trading rounds, K sequential sized offers per round, a Bureau de Change
 choice each round once the rate is revealed, settlement at the round-(n+1)
 rate with the residue and deficit penalties. rounds and K are the knobs

 --- Conventions ---------------------------------------------------------------
   spec.side   WHO the trader is: "A" starts pounds / targets dollars,
               "B" starts dollars / targets pounds. Does NOT restrict
               which way a trade may run.
   direction   which way THIS trade runs, chosen fresh each step by the SIGN
               of the size action: "A" sells pounds, "B" sells dollars.
               fx_mechanics takes the direction, never spec.side.
   env holdings    self.c / self.d are PHYSICAL: always pounds / dollars.
   solver holdings terminal_wealth and the DP take (c, d) as ROLES: initial-
               then target-currency. Same thing on side A, SWAPPED on side B
               -- every call into the solver goes through the (held, banked)
               reordering, and skipping it is silent on A and wrong by a
               factor of a_end^2 on B.

 -------0-------------------------------------------------------------------
 One episode = one game, rounds*(K+1) steps: K offer-steps (X hidden), one
 BdC-step (X revealed). 

 Observation (8 floats), all continuous, absolute units, no model parameters:
     0  rounds_left / rounds          (time left across rounds)
     1  offers_left / K               (offers left this round; 0 at the BdC)
     2  phase                         (0 = offer step, 1 = BdC step)
     3  initial-currency holding / L  (side A: c/L;  side B: d/L)
     4  target-currency holding  / T  (side A: d/T;  side B: c/T)
     5  last revealed rate, absolute  (round 1: the published opening rate)
     6  bracket floor on X, absolute  (P_MIN when nothing learned yet)
     7  bracket ceiling on X, absolute (P_MAX when nothing learned yet;
                                       both collapse to X at the BdC step)

 Action (2 floats):
     offer step:  a[0] -> THE PRICE, absolute rate units, clipped to
                          [P_MIN, P_MAX], quoted $/GBP (the convention X is
                          stated in). 
                  a[1] -> SIGNED FRACTION, continuous on [-1, +1]: sign =
                          trade direction, magnitude = fraction of the
                          relevant holding. +0.37 sells 37% of the pounds;
                          -0.60 sells 60% of the dollars; 0 offers nothing
                          (the accept/reject bit still arrives).
     BdC step:    a[1] -> SIGNED dump, same convention: |m| is the fraction
                          banked at the revealed rate; m = 0 carries; any
                          partial dump is allowed.
                  a[0] -> ignored
================================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field

from Mechanics.fx_mechanics import GameParams, mm_accepts, trade_rate, bdc_payoff_per_unit, A0_DEFAULT
from DP_Models.v2_multiple_rounds import TraderSpec, terminal_wealth

P_MIN, P_MAX = 0.0, 8.0 * A0_DEFAULT      

@dataclass
class GameSpec(TraderSpec):
    L: float = 100000.0          # starting capital, INITIAL currency
    T: float = 125000.0          # target, TARGET currency
    A: float = 0.02              # residue penalty on leftover initial currency
    B: float = 0.03               # deficit penalty on the target shortfall below T
    rounds: int = 4               # trading rounds (round n+1 = settlement draw only)
    K: int = 1                   # sequential offers per round; 1 = one guess
    side: str = "A"               # "A": pounds -> dollars ; "B": reverse 
    params: GameParams = field(default_factory=GameParams)


def _card(side, **over):
    """Gate plumbing: the two cards' numbers, overridable per gate."""
    base = (dict(L=100_000.0, T=125_000.0, A=0.02, B=0.03, side="A")
            if side == "A" else
            dict(L=500_000.0, T=396_000.0, A=0.00, B=0.20, side="B"))
    base.update(over)
    return GameSpec(**base)


class Game(gym.Env):
    """The trading game as a Gymnasium environment (continuous, no grids,
    either side, either direction)."""

    def __init__(self, spec: TraderSpec = None):
        super().__init__()
        self.spec = spec or GameSpec()
        p = self.spec.params
        self.a0, self.sd, self.f = p.a0, p.sd, p.bdc_fee

        obs_low = np.array([0, 0, 0, 0, 0, 0, P_MIN, P_MIN],
                           dtype=np.float32)
        obs_high = np.array([1, 1, 1, np.inf, np.inf, np.inf, P_MAX, P_MAX],
                            dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        act_low = np.array([P_MIN, -1.0], dtype=np.float32)
        act_high = np.array([P_MAX, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        
        # Every piece of episode state is created HERE, so no method can ever
        # depend on reset() having run first. reset() overwrites these.
        self.rng = np.random.default_rng()       # reseeded by reset(seed=...)
        self._force_path = self._force_end = None
        self.W = self.a_end = None
        self.n, self.k, self.phase = 1, self.spec.K, 0
        self.a, self.X = self.a0, self.a0
        self.lo, self.hi = -np.inf, np.inf
        self.c, self.d = ((self.spec.L, 0.0) if self.spec.side == "A"
                          else (0.0, self.spec.L))

    # ---- one trade, either way ----------------------------------------------
    def _spend(self, signed_frac):
        """Decode the size action into (direction, q).

        signed_frac  the raw action slot, in [-1, +1]: SIGN picks the trade
                     direction, |magnitude| is the fraction of the relevant
                     holding to sell.
        q            the resulting AMOUNT of the sold currency  

        Direction "A" sells pounds, "B" sells dollars."""
        if signed_frac >= 0.0:
            return "A", signed_frac * self.c
        return "B", (-signed_frac) * self.d

    def _apply(self, direction, q, rate_per_unit):
        """Move q units of the sold currency at `rate_per_unit` units of the
        bought currency each. One path for MM fills and BdC dumps, and for
        both directions."""
        recv = q * rate_per_unit
        if direction == "A":
            self.c -= q
            self.d += recv
        else:
            self.d -= q
            self.c += recv
        return recv

    # ---- observation ---------------------------------------------------------
    def _obs(self):
        s = self.spec
        held, banked = (self.c, self.d) if s.side == "A" else (self.d, self.c)
        rounds_left = (s.rounds - self.n + 1) / s.rounds
        a_obs = self.a
        if self.phase == 0:                      # offers: X hidden
            lo_f = (P_MIN if self.lo == -np.inf
                    else float(np.clip(self.lo, P_MIN, P_MAX)))
            hi_f = (P_MAX if self.hi == np.inf
                    else float(np.clip(self.hi, P_MIN, P_MAX)))
        else:                                    # BdC: X revealed
            lo_f = hi_f = float(np.clip(self.X, P_MIN, P_MAX))
        return np.array([rounds_left, self.k / s.K, float(self.phase),
                         held / s.L, banked / s.T, a_obs, lo_f, hi_f],
                        dtype=np.float32)

    # ---- round setup ---------------------------------------------------------
    def _start_round(self):
        self.k = self.spec.K
        self.phase = 0
        self.lo, self.hi = -np.inf, np.inf
        if self._force_path is not None:         # validation only
            self.X = float(self._force_path[self.n - 1])
        else:
            self.X = self.a + self.sd * self.rng.standard_normal() #random normal evolution

    # Starts a new episode
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # optional forced rates: options={"path": [X_1..X_rounds], "a_end": ...}
        self._force_path = options.get("path") if options else None
        self._force_end = options.get("a_end") if options else None
        self.n = 1
        self.a = self.a0
        self.c, self.d = ((self.spec.L, 0.0) if self.spec.side == "A"
                          else (0.0, self.spec.L))
        self._start_round()
        return self._obs(), {}

    # ---- one step ------------------------------------------------------------
    def step(self, action):
        s = self.spec
        reward, terminated = 0.0, False

        if self.phase == 0:
            # --- one MM offer: the agent's own price, in its signed direction ---
            P = float(np.clip(action[0], P_MIN, P_MAX))
            signed_frac = float(np.clip(action[1], -1.0, 1.0))
            # q is OFFERED, not yet traded: on rejection nothing moves and
            # only the bracket updates (the DP's acceptance branch goes to
            # (c - q, d + P*q), its rejection branch keeps holdings)
            if abs(signed_frac) <= 1e-8:
                # Stop: no offer, no accept/reject information.
                self.k = 0
                self.phase = 1
            else:
                direction, q = self._spend(signed_frac)
                if mm_accepts(P, self.X, direction):
                    self._apply(direction, q, trade_rate(P, direction))
                    # a fill puts P on the trader's side of X: selling pounds
                    # learns X > P (floor), selling dollars learns X < P (ceiling)
                    if direction == "A":
                        self.lo = max(self.lo, P)
                    else:
                        self.hi = min(self.hi, P)
                else:                                # a rejection says the opposite
                    if direction == "A":
                        self.hi = min(self.hi, P)
                    else:
                        self.lo = max(self.lo, P)
                self.k -= 1
                if self.k == 0:
                    self.phase = 1                   # next: BdC (reveals X)
        else:
            # --- the BdC choice at the now-revealed rate X ---
            signed_frac = float(np.clip(action[1], -1.0, 1.0))
            direction, m = self._spend(signed_frac)      # m: amount dumped
            self._apply(direction, m,
                        bdc_payoff_per_unit(self.X, self.f, direction))
            self.a = self.X                      # revealed rate becomes the anchor
            self.n += 1
            if self.n > s.rounds:                # settlement
                self.a_end = (float(self._force_end)
                              if self._force_end is not None
                              else self.a + self.sd * self.rng.standard_normal())
                # the solver's terminal_wealth takes ROLE order (initial,target)
                held, banked = ((self.c, self.d) if s.side == "A"
                                else (self.d, self.c))
                self.W = terminal_wealth(s, held, banked, self.a_end)
                reward = (self.W - s.L) / s.L    # P/L as a fraction of capital
                terminated = True
            else:
                self._start_round()

        if terminated:
            held, banked = ((self.c, self.d) if s.side == "A"
                            else (self.d, self.c))
            obs = np.array([0, 0, 1, held / s.L, banked / s.T,
                            self.a, 0.0, 0.0], dtype=np.float32)
        else:
            obs = self._obs()
        info = {"W": self.W, "a_end": self.a_end} if terminated else {}
        return obs, reward, terminated, False, info


# ==============================================================================
# VALIDATION -- every reference number computed at run time, none hard-coded
# ==============================================================================

def _play(env, policy, seed=None, options=None):
    """Play one episode with a callable policy(obs, env) -> action."""
    obs, _ = env.reset(seed=seed, options=options)
    done, total = False, 0.0
    while not done:
        obs, r, done, _, _ = env.step(policy(obs, env))
        total += r
    return total, env.c, env.d


def gate_G1(side="A", n_games=2_000, seed=2, print_progress=True):
    """Structure and conservation on the FULL card (K=3, rounds=4, penalties
    live) under RANDOM SIGNED actions, so both directions fire: holdings never
    negative; every trade moves value at exactly the traded rate; the bracket
    always contains X at the offer steps; episode length = rounds*(K+1); and
    the reward equals terminal_wealth recomputed independently."""
    env = Game(_card(side, K=3, rounds=4))
    rng = np.random.default_rng(seed)
    both = [0, 0]
    for _ in range(n_games):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        steps, done, r, info = 0, False, 0.0, {}
        while not done:
            c0, d0, phase, X, a = env.c, env.d, env.phase, env.X, env.a
            if phase == 0:
                assert env.lo < X <= env.hi + 1e-12, "bracket lost X"
            act = np.array([rng.uniform(P_MIN, P_MAX),
                            rng.uniform(-1.0, 1.0)], dtype=np.float32)
            obs, r, done, _, info = env.step(act)
            assert env.observation_space.contains(obs), "obs outside declared box"
            steps += 1
            assert env.c >= -1e-6 and env.d >= -1e-6, "negative holdings"
            dc, dd = env.c - c0, env.d - d0
            if abs(dc) > 1e-9:
                both[0 if dc < 0 else 1] += 1
                rate = -dd / dc
                ref = (float(np.clip(act[0], P_MIN, P_MAX))
                       if phase == 0 else
                       ((1 - env.f) * X if dc < 0 else X / (1 - env.f)))
                assert abs(rate - ref) < 1e-6 * max(1.0, abs(ref)), "rate mismatch"
        assert steps == env.spec.rounds * (env.spec.K + 1), "episode length wrong"
        sp = env.spec
        held, banked = (env.c, env.d) if sp.side == "A" else (env.d, env.c)
        r_end = 1.0 / info["a_end"] if sp.side == "A" else info["a_end"]
        W_ind = held * (1 - sp.A) + (banked
                                     - sp.B * max(sp.T - banked, 0.0)) * r_end
        assert abs(r - (W_ind - sp.L) / sp.L) < 1e-9, "reward != wealth eqn"
    if print_progress:
        print(f"  side {side}: {n_games} random games (K=3, rounds=4), "
              f"{both[0]} pound-sales / {both[1]} dollar-sales, all checks ok")
    return both[0] > 0 and both[1] > 0


def gate_G2(side="A", n_games=20_000, seed=5, print_progress=True):
    """The two-way plumbing, pinned exactly. A round trip through the BdC --
    sell the whole book in round 1, buy it all back in round 2 -- returns
    (1-f)^2 * (rate ratio) per unit: the legs are reciprocal apart from the
    fee and apart from the rate moving between them. Forcing a flat path
    isolates the fee, so with penalties off the P/L is exactly (1-f)^2 - 1 on
    every path, zero variance -- the domination result as a unit test."""
    spec = _card(side, rounds=2, K=1, A=0.0, B=0.0)   # isolate the fee
    env = Game(spec)
    a0, sd, f = spec.params.a0, spec.params.sd, spec.params.bdc_fee
    out, back = (1.0, -1.0) if side == "A" else (-1.0, 1.0)

    def pol(obs, env):
        if env.phase == 0:
            return np.array([1.0, 0.0], dtype=np.float32)   # offer nothing
        return np.array([1.0, out if env.n == 1 else back], dtype=np.float32)

    rng = np.random.default_rng(seed)
    X = a0 + sd * rng.standard_normal(n_games)
    flat = np.array([_play(env, pol, options={"path": [X[i], X[i]],
                                              "a_end": X[i]})[0]
                     for i in range(n_games)])
    expected = (1 - f) ** 2 - 1.0
    ok = abs(flat.mean() - expected) < 1e-12 and flat.std() < 1e-12
    if print_progress:
        print(f"  side {side} flat-path round trip P/L {flat.mean():+.8f}  "
              f"(exact (1-f)^2-1 = {expected:+.8f}, sd {flat.std():.2e})"
              f"  [{'ok' if ok else 'FAIL'}]")
    return ok


def gate_G3(side="A", rounds=4, n_games=20_000, seed=7, print_progress=True):
    """THE anchor: the v2 DP solves this exact objective, so its optimal
    policy replayed through THIS env must return the DP's own value. Two
    independent implementations of the same game, checked against each other
    -- Avi's value-versus-strategy consistency test (section 3.4), which is
    a real test precisely because value and strategy are distinct objects.
    K = 1, penalties live at the card, so nothing is switched off."""
    from DP_Models.v2_multiple_rounds import (solve_v2, greedy_offer_action,
                                    greedy_bdc_action)
    spec = _card(side, rounds=rounds, K=1)
    sol = solve_v2(spec)
    env = Game(spec)
    grids = sol.grids

    from DP_Models.v2_multiple_rounds import NO_FLOOR, _rate_at

    def _z_idx(P, a, upper):
        """Map a realised price back onto the DP's offer-index grid. The
        bracket the env carries in RATE units is the same object the DP
        carries as an index; snap conservatively so the replay never claims
        to know more than the DP does."""
        z = (P - a) / spec.params.sd if spec.side == "A" else (a - P) / spec.params.sd
        j = np.searchsorted(grids.z, z)
        return int(np.clip(j, 0, grids.n_offer)) if upper else \
               int(np.clip(j - 1, 0, grids.n_offer - 1))

    def pol(obs, env):
        if env.phase == 0:
            if spec.side == "A":
                lo = NO_FLOOR if env.lo == -np.inf else _z_idx(env.lo, env.a, False)
                hi = grids.n_offer if env.hi == np.inf else _z_idx(env.hi, env.a, True)
            else:   # buyer: the reflection swaps which bound is the DP's floor
                lo = NO_FLOOR if env.hi == np.inf else _z_idx(env.hi, env.a, False)
                hi = grids.n_offer if env.lo == -np.inf else _z_idx(env.lo, env.a, True)
            held, banked = ((env.c, env.d) if spec.side == "A"
                            else (env.d, env.c))      # role order, as the DP
            act, _ = greedy_offer_action(sol, env.n, env.k, lo, hi,
                                         held, banked, env.a)
            if act is None:                      # DP says stop: offer nothing
                return np.array([1.0, 0.0], dtype=np.float32)
            p_idx, q = act
            P = _rate_at(spec, env.a, grids.z[p_idx])
            # q is an amount of the INITIAL currency; the env wants a signed
            # fraction, negative meaning "sell dollars"
            # the DP hands back an AMOUNT q; the env's action slot wants a
            # signed fraction of the same holding
            signed_frac = ((q / held if held > 0 else 0.0)
                           * (1 if spec.side == "A" else -1))
            return np.array([P, signed_frac], dtype=np.float32)
        held, banked = ((env.c, env.d) if spec.side == "A"
                        else (env.d, env.c))
        m = greedy_bdc_action(sol, env.n, held, banked, env.X)   # an amount
        signed_frac = ((m / held if held > 0 else 0.0)
                       * (1 if spec.side == "A" else -1))
        return np.array([1.0, signed_frac], dtype=np.float32)

    rng = np.random.default_rng(seed)
    pay = np.array([_play(env, pol, seed=int(rng.integers(1 << 31)))[0]
                    for _ in range(n_games)])
    dp_pl = sol.pl()
    se = pay.std(ddof=1) / np.sqrt(n_games)
    gap = abs(pay.mean() - dp_pl)
    ok = gap < 4 * se
    if print_progress:
        print(f"  side {side} rounds={rounds} K=1: env replay P/L "
              f"{pay.mean():+.6f} +/- {2*se:.6f}   DP value {dp_pl:+.6f}"
              f"   gap {gap:.2e}  [{'ok' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    res = {}
    for side in ("A", "B"):
        print(f"===== side {side} " + "=" * 45)
        print("gate G1: structure + conservation, full card, random signed play")
        res[("G1", side)] = gate_G1(side)
        print("gate G2: two-way round trip costs exactly (1-f)^2")
        res[("G2", side)] = gate_G2(side)
        print("gate G3: v2 DP policy replayed here matches the DP value")
        res[("G3", side)] = gate_G3(side)
    print("=" * 60)
    for side in ("A", "B"):
        print(f"side {side}   " + "  ".join(
            f"{g}:{'pass' if res[(g, side)] else 'FAIL'}"
            for g in ("G1", "G2", "G3")))