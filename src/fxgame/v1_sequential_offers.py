"""
================================================================================
 VERSION 1
 One liquidator, one round, up to K sequential all-or-nothing offers,
 hidden true rate.
================================================================================

 Structure:
   1. BELIEFS      - the bell curve, and its two truncated-normal facts
   2. THE SOLVER   - backward induction on a price grid (+ V1Solution readers)
   3. MONTE CARLO  - play the policy for real (via shared rules), as a check
   4. VALIDATION   - K=1 must reproduce v0; DP must match its own Monte Carlo
   5. PLOT         - value vs K, and the descending offer ladder
   6. THE LADDERS  - the optimal strategy read off the policy table
   7. RUN          - headline table, ladders, both validations, the figure

================================================================================
"""

import random
from scipy.stats import norm      # Phi = norm.cdf, phi = norm.pdf (same as v0)
import matplotlib.pyplot as plt

# The rules of the game, defined once and shared with v0. See fx_mechanics.py.
from fx_mechanics import (
    GameParams,
    mm_accepts,
    bdc_payoff_per_pound,
    A0_DEFAULT,
    sd_DEFAULT,
    BDC_FEE_DEFAULT,
)


# ==============================================================================
# 1. BELIEFS
# ==============================================================================
#
# We normalise the price with respect to the last revealed rate":
#
#       z_y = (y - a0) / sd
#
# The two facts the solver needs about the truncated normal (belief after a
# rejection has capped X at the ceiling c) both come straight from the normal:
#
#   (i)  Probability an offer P is accepted, given the ceiling c (P < c):
#
#            Pr(X > P | X <= c) = (Phi(z_c) - Phi(z_P)) / Phi(z_c)
#
#
#
#   (ii) Average rate given it is below the ceiling (the BdC fallback needs it):
#
#            E[X | X <= c] = a0 - sd * phi(z_c) / Phi(z_c)
#
#        
#
# These are written out inline in the solver where they are used.


# ==============================================================================
# 2. THE SOLVER
# ==============================================================================

class V1Solution:
    """
    grid[i]          the i-th price on the shared grid.

    values[k][j]     V_k(ceiling = grid[j]) -- how well you can do from here.

    policy[k][j]     grid index of the best next offer when k offers remain and
                     the ceiling is grid[j].

    start_value[k]   value of the round with k offers and NO ceiling yet -- the
                     headline number (V_k(+∞)).

    start_index[k]   grid index of the best FIRST offer with k offers.

    params           the GameParams this was solved for.
    """

    def __init__(self, grid, values, policy, start_value, start_index, params):
        self.grid = grid
        self.values = values
        self.policy = policy
        self.start_value = start_value
        self.start_index = start_index
        self.params = params

    def value(self, K):
        """Expected dollars-per-pound with K offers, before anything happens."""
        return self.start_value[K]

    def first_offer(self, K):
        """The best opening offer with K offers in hand."""
        return self.grid[self.start_index[K]]

    def ladder(self, K):
        """The offers made if EVERY offer is rejected."""
        i = self.start_index[K]
        rungs = [self.grid[i]]
        for k in range(K - 1, 0, -1):     # after a rejection the ceiling is i
            i = self.policy[k][i]
            if i is None:
                break
            rungs.append(self.grid[i])
        return rungs


def solve_v1(params=None, max_offers=10, grid_halfwidth_sds=6.0, n_grid=801):
    """Backward-induction solution of the K-sequential-offer round.

    The grid covers a0 ± grid_halfwidth_sds standard deviations with n_grid
    points. Because the objective is flat at its maximum, snapping to the
    nearest grid point barely moves the value: at K=1 the grid value matches
    v0's closed form to ~1e-8
    """
    if params is None:
        params = GameParams()
    a0, sd, f = params.a0, params.sd, params.bdc_fee
    n = n_grid

    # ---- the shared price grid, with the norm evaluated on it once ----
    step = 2.0 * grid_halfwidth_sds / (n - 1)
    z = [-grid_halfwidth_sds + step * i for i in range(n)]   # z-scores
    grid = [a0 + sd * zi for zi in z]        # grid[i]: i-th possible price
    Phi = [norm.cdf(zi) for zi in z]            # Phi[i] = Pr(X <= grid[i])
    phi = [norm.pdf(zi) for zi in z]

    # ---- V_0: out of offers, wait for the BdC -------------------------------
    # (1 - f) * E[X | X <= ceiling].
    V_prev = [(1.0 - f) * (a0 - sd * phi[j] / Phi[j]) for j in range(n)]

    values = [V_prev]               # values[k][j] = V_k(ceiling = grid[j])
    policy = [[None] * n]           # k = 0: no decision to make
    start_value = [None]            # index 0 unused: k starts at 1
    start_index = [None]

    for k in range(1, max_offers + 1):

        # ---- (a) the FIRST offer of the round: no ceiling yet ---------------
        # Accepted (prob 1 - Phi[i]): sold at grid[i].
        # Rejected (prob Phi[i]):     ceiling becomes grid[i], play on with
        #                             k-1 offers -> value V_{k-1}(grid[i]).
        # For k = 1 this is v0's g(P).
        best_value = float("-inf")
        best_i = 0
        for i in range(n):
            payoff = grid[i] * (1.0 - Phi[i]) + Phi[i] * V_prev[i]
            if payoff > best_value:
                best_value = payoff
                best_i = i
        start_value.append(best_value)
        start_index.append(best_i)

        # ---- (b) V_k(c) for every possible ceiling c on the grid ------------
        V_new = [0.0] * n
        pol = [None] * n
        for j in range(n):                      # j indexes the ceiling grid[j]
            best_value = values[0][j]
            best_i = None
            for i in range(j):
                p_accept = (Phi[j] - Phi[i]) / Phi[j]     # fact (i)
                p_reject = Phi[i] / Phi[j]
                payoff = grid[i] * p_accept + p_reject * V_prev[i]
                if payoff > best_value:
                    best_value = payoff
                    best_i = i
            V_new[j] = best_value
            pol[j] = best_i

        values.append(V_new)
        policy.append(pol)
        V_prev = V_new

    return V1Solution(grid, values, policy, start_value, start_index, params)


# ==============================================================================
# 3. MONTE CARLO  (independent check: play the policy for real, many times)
# ==============================================================================

def monte_carlo_value(sol, K, n_games=200_000, seed=0):
    """Simulate the round exactly as a player would live it, and average the
    result. This must agree with sol.value(K): same two-way discipline as v0
    (one formula, one simulation).
    """
    if K >= len(sol.start_value):
        raise ValueError(f"K={K} exceeds solved budget (max {len(sol.start_value)-1}); "
                         f"re-run solve_v1 with a larger max_offers.")
    p = sol.params
    rng = random.Random(seed)
    total = 0.0
    total_sq = 0.0
    for _ in range(n_games):
        X = rng.gauss(p.a0, p.sd)               # the hidden rate, drawn once
        i = sol.start_index[K]                  # the opening offer
        payoff = None
        for k in range(K, 0, -1):
            offer_rate = sol.grid[i]
            if mm_accepts(offer_rate, X):   
                payoff = offer_rate             # sold everything at the offer
                break
            if k > 1:                           
                i = sol.policy[k - 1][i]        #Update index to the next optimal value using k and the new ceiling
                if i is None:                   
                    break
        if payoff is None:                      # every offer was rejected
            payoff = bdc_payoff_per_pound(X, p.bdc_fee)   
        total += payoff
        total_sq += payoff * payoff

    mean = total / n_games
    variance = total_sq / n_games - mean * mean
    std_error = (max(variance, 0.0) / n_games) ** 0.5
    return mean, std_error


# ==============================================================================
# 4. VALIDATION
# ==============================================================================
from v0_game import closed_form_optimal_rate, expected_dollars_per_pound

def test_k1_matches_v0():
    """
    With one offer and no ceiling, v1's recursion collapses to v0's g(P), so
    the solver's K=1 answer must reproduce v0's closed form -- imported directly
    from v0_game.py, which is itself validated against a Monte Carlo there."""

    sol = solve_v1(max_offers=1)
    P_star = closed_form_optimal_rate(A0_DEFAULT, sd_DEFAULT, BDC_FEE_DEFAULT)
    value = expected_dollars_per_pound(P_star, A0_DEFAULT, sd_DEFAULT, BDC_FEE_DEFAULT)
    assert abs(sol.value(1) - value) < 1e-4, "K=1 value disagrees with v0"
    assert abs(sol.first_offer(1) - P_star) < 2e-3, "K=1 offer disagrees with v0"


def test_monte_carlo_matches_dp():
    """The DP value must match a simulation of its own policy, played through
    the shared game rules. This IS the independent check for v1: a different
    method (Monte Carlo) against a different method (backward induction), both
    testing v1's own solution. Checks a couple of budgets within 4 std errors."""
    sol = solve_v1(max_offers=10)
    for K in (1, 5, 10):
        mc, se = monte_carlo_value(sol, K, n_games=200_000, seed=1)
        assert abs(mc - sol.value(K)) < 4 * se, f"MC disagrees with DP at K={K}"



# ==============================================================================
# 5. PLOT
# ==============================================================================

def plot(sol, ladder_K=5, save_path="v1_plot.png"):
    """Two panels: (left) value climbing with K towards the perfect-information
    limit; (right) the descending offer ladder if every offer is rejected."""
    p = sol.params
    a0, f = p.a0, p.bdc_fee
    Ks = list(range(1, len(sol.start_value)))
    vals = [sol.value(k) for k in Ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: the fee being clawed back, offer by offer.
    ax1.plot(Ks, vals, marker="o", lw=2, color="tab:blue",
             label="optimal value with K offers")
    ax1.axhline(a0, ls="--", color="grey",
                label=f"perfect information (avg) = {a0:.4f}")
    ax1.axhline((1 - f) * a0, ls=":", color="grey",
                label=f"BdC only = {(1 - f) * a0:.4f}")
    ax1.set_xlabel("K (offers allowed in the round)")
    ax1.set_ylabel("Expected dollars per pound")
    ax1.set_title("The whole prize is the fee: value vs K")
    ax1.set_xticks(Ks)
    ax1.legend(fontsize=8)

    # Right: what price discovery looks like on the worst path.
    rungs = sol.ladder(ladder_K)
    ax2.step(range(1, len(rungs) + 1), rungs, where="mid",
             marker="o", lw=2, color="tab:orange")
    ax2.axhline(a0, ls="--", color="grey", label=f"a0 = {a0:.2f}")
    ax2.set_xlabel("offer number (each one rejected)")
    ax2.set_ylabel("offer price ($ per pound)")
    ax2.set_title(f"K = {ladder_K}: offers if every one is rejected")
    ax2.set_xticks(range(1, len(rungs) + 1))
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ==============================================================================
# 6. THE LADDERS  (the optimal strategy, read straight off the policy table)
# ==============================================================================
 
def print_ladders(sol, max_K=10):
    """Print, for each K = 1..max_K, the full sequence of offers the optimal
    strategy makes IF EVERY OFFER IS REJECTED -- the 'all-rejected ladder'.
    """

    a0 = sol.params.a0
    print(f"Optimal offer ladders (each row = offers made if all are rejected)")
    print(f"Read a row until the first offer < X; that is where it is accepted.")
    print(f"(last revealed rate a0 = {a0:.4f})\n")
    print(f"{'K':>3}  offers, high to low  (each one rejected -> next)")
    for K in range(1, max_K + 1):
        rungs = sol.ladder(K)
        row = "  ".join(f"{r:.4f}" for r in rungs)
        print(f"{K:>3}  {row}")

# ==============================================================================
# 7. Run everything
# ==============================================================================

if __name__ == "__main__":
    params = GameParams()               # the base game: a0=1.25, sd=0.05, f=0.02
    sol = solve_v1(params, max_offers=10)
    a0, f = params.a0, params.bdc_fee

    # ---- headline table ------------------------------------------------------
    # "% of fee pie": perfect information earns a0 on average; the BdC earns
    # (1-f)*a0; the gap f*a0 is the whole prize. This shows how much K offers
    # recover.
    pie = f * a0
    print(f"{'K':>3} {'value $/GBP':>12} {'first offer':>12} {'% of fee pie':>13}")
    for k in range(1, 11):
        share = 100.0 * (sol.value(k) - (1 - f) * a0) / pie
        print(f"{k:>3} {sol.value(k):>12.5f} {sol.first_offer(k):>12.4f} "
              f"{share:>12.1f}%")

    # ---- the full ladders for every K (the optimal strategy) -------------
    print()
    print_ladders(sol, max_K=10)

    # ---- validation 1: K = 1 is v0 -------------------------------------------
    P_star = closed_form_optimal_rate(a0, params.sd, f)
    v0_value = expected_dollars_per_pound(P_star, a0, params.sd, f)
    print(f"\nK=1 check vs v0:  offer {sol.first_offer(1):.4f} "
          f"(v0 P* = {P_star:.4f}),  value {sol.value(1):.5f} "
          f"(v0 = {v0_value:.5f})")
    test_k1_matches_v0()
    print("K=1 reproduces v0: PASSED")

    # ---- validation 2: the dynamic programming vs a Monte Carlo of its own policy -------------
    # A fresh random sample each run; the DP values are exact, so this is a live
    # confirmation that a simulation of the policy lands within error of them
    for K in (1, 5, 10):
        mc, se = monte_carlo_value(sol, K, n_games=200_000, seed = None)
        print(f"K={K:>2}: DP value {sol.value(K):.5f}   "
              f"Monte Carlo {mc:.5f} +/- {2 * se:.5f} (2 s.e.)")

    # ---- the figure ---------------------------------------------------
    plot(sol, ladder_K=5)
