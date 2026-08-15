import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
v0_sigma_sweep.py -- Avi's experiment: how the one-shot optimal offer moves
                     with the rate volatility sigma.

Everything is the v0 closed form imported from v0_game -- nothing re-derived
here, nothing cached; every run solves fresh. Units come from fx_mechanics,
so each side is drawn the way its own trader would quote it: side A in
$/GBP, side B in GBP/$.

WHAT IT SHOWS: the optimal offer has TWO REGIMES either side of a critical
volatility sigma_c.
  * sigma > sigma_c  FISHING: quote away from the anchor in the greedy
                     direction, hoping the hidden rate ran the trader's way.
                     Fills are rare and rich.
  * sigma < sigma_c  FEE-DODGING: quote on the OTHER side of the anchor. The
                     hidden rate is nearly known, so an offer a touch worse
                     than fair value fills almost surely and beats paying the
                     Bureau's fee. The market maker stops mattering and the
                     offer's only job is avoiding the fee.
The crossing is exact: h(a0) = 0 gives 1/2 = f*a0/(sigma*sqrt(2*pi)), so
        sigma_c = sqrt(2/pi) * f * a0,
printed below for three fees and both sides.

A caution the plot respects: where the offer channel is DOMINATED there is no
interior optimum at all, and the finder correctly returns a bracket edge.
Drawing that edge would put a bracket setting on the page as though it were
economics, so those points are left off the line.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from v0_game import (closed_form_optimal_rate, expected_target_per_unit, h,
                     A0_DEFAULT, BDC_FEE_DEFAULT)
# the trader's own units: side A quotes $/GBP, side B quotes GBP/$ (= 1/P)
from Mechanics.fx_mechanics import trade_rate, rate_units, results_path

# ============================ SETTINGS ======================================
SIGMA_LO, SIGMA_HI = 0.002, 0.15   # sweep range (the card sits at 0.05)
N_SIGMA = 400                   # points on the sweep
FEES = (0.01, 0.02, 0.05)       # sigma_c printed for each of these
FEE_PLOT = BDC_FEE_DEFAULT      # the card fee (2%), used for the figure
CHECK_SIGMAS = (0.005, 0.05, 0.12)  # argmax cross-check points
N_CHECK = 2_001                 # grid for that check. Kept modest because
                                # side B's objective is an adaptive quadrature
                                # per point, so a fine grid costs minutes.
SAVE = "v0_sigma_sweep.png"
# ============================================================================

a0 = A0_DEFAULT
SIDES = ("A", "B")
SIDE_NAME = {"A": "side A   pounds " + chr(8594) + " dollars",
             "B": "side B   dollars " + chr(8594) + " pounds"}


def p_star(sigma, fee, side):
    return closed_form_optimal_rate(a0, sigma, fee, side=side)


def sigma_crit(fee, side):
    """The volatility at which the optimal offer crosses the anchor."""
    return brentq(lambda s: p_star(s, fee, side) - a0, 1e-4, SIGMA_HI)


def is_interior(P, sigma, fee, side):
    """Is P a genuine stationary point, or a bracket edge?

    The test must be RELATIVE. The first-order condition is (probability of a
    fill) - (marginal cost of shading), and eight sd from the anchor BOTH
    terms are around 1e-16, so their difference passes any absolute tolerance
    and a boundary impersonates a root. Scaling by the size of the terms
    themselves separates them: the ratio is 0 at a real optimum, O(1) at an
    edge."""
    cost = fee * (P / sigma) * norm.pdf((P - a0) / sigma)
    gap = h(P, a0, sigma, fee, side)                  # fill - cost
    return abs(gap) / max(gap + 2.0 * cost, 1e-300) < 1e-6


if __name__ == "__main__":
    print("=" * 74)
    print(f"v0 SIGMA SWEEP  |  a0 = {a0}, figure at fee {FEE_PLOT:.0%}")
    print("=" * 74)

    print("\nthe critical volatility sigma_c (the offer crosses the anchor):")
    for side in SIDES:
        for fee in FEES:
            sc = sigma_crit(fee, side)
            print(f"  side {side}  fee {fee:4.0%}:  sigma_c = {sc:.4f}"
                  f"   (sigma_c / (fee*a0) = {sc / (fee * a0):.3f}"
                  f",  sqrt(2/pi) = {np.sqrt(2 / np.pi):.3f})")

    print(f"\nargmax cross-check (closed form vs {N_CHECK:,}-point grid).")
    print("  The verdict is on the VALUE achieved, not the location: where the")
    print("  offer channel is dominated the objective is monotone, so the two")
    print("  agree on 'never fill' while sitting at different prices.")
    for side in SIDES:
        for s in CHECK_SIGMAS:
            Ps = p_star(s, FEE_PLOT, side)
            grid = np.linspace(max(1e-6, a0 - 8 * s), a0 + 8 * s, N_CHECK)
            vals = np.array([expected_target_per_unit(P, a0, s, FEE_PLOT,
                                                      side=side)
                             for P in grid])
            g_star = expected_target_per_unit(Ps, a0, s, FEE_PLOT, side=side)
            ok = g_star >= vals.max() - 1e-9
            print(f"  side {side}  sigma {s:5.3f}:  closed form {Ps:.5f} "
                  f"(g {g_star:.8f})   grid best {grid[int(np.argmax(vals))]:.5f} "
                  f"(g {vals.max():.8f})  [{'ok' if ok else 'WORSE THAN GRID'}]")

    # -- the sweep and the figure: side A left, side B right -----------------
    sig = np.linspace(SIGMA_LO, SIGMA_HI, N_SIGMA)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    print()
    for ax, side in zip(axes, SIDES):
        P = np.array([p_star(s, FEE_PLOT, side) for s in sig])
        interior = np.array([is_interior(Pi, si, FEE_PLOT, side)
                             for Pi, si in zip(P, sig)])
        if not interior.all():
            print(f"  side {side}: no interior optimum for sigma >= "
                  f"{sig[~interior].min():.4f} (offer channel dominated -- "
                  f"Bureau only); those points are left off the line")
        offer = np.where(interior, [trade_rate(Pi, side) for Pi in P], np.nan)
        anchor = trade_rate(a0, side)

        ax.plot(sig, offer, lw=2, color="#1a5276")
        ax.axhline(anchor, ls="--", lw=1, color="0.55",
                   label=f"anchor = {anchor:.4f}")
        ax.set_xlabel(r"rate volatility $\sigma$")
        ax.set_ylabel(f"optimal offer  [{rate_units(side)}]")
        # a shared x range, so where a line STOPS is visible rather than
        # hidden by autoscaling to the surviving points
        ax.set_xlim(SIGMA_LO, SIGMA_HI)
        ax.set_title(SIDE_NAME[side], fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = results_path(SAVE)
    fig.savefig(out, dpi=170)
    print(f"\nwrote {SAVE}   (sigma_c = {sigma_crit(FEE_PLOT, 'A'):.4f} at the "
          f"card fee; the card's sigma = 0.05 is in the fishing regime)")