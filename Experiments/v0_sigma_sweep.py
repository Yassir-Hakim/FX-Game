import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from DP_Models.v0_game import (closed_form_optimal_rate, offer_is_interior,
                     A0_DEFAULT, BDC_FEE_DEFAULT)
# the trader's own units: side A quotes $/GBP, side B quotes GBP/$ (= 1/P)
from Mechanics.fx_mechanics import trade_rate, rate_units, results_path

# ============================ SETTINGS ======================================
SIGMA_LO, SIGMA_HI = 0.002, 0.15   # sweep range (the card sits at 0.05)
N_SIGMA = 400                      # points on the sweep
FEE = BDC_FEE_DEFAULT              # the card fee (2%) -- the only fee here
SHOW_RECIPROCAL = False             # dashed 1/P*_A reference on the B panel
SAVE = "v0_sigma_sweep.png"
# ============================================================================

a0 = A0_DEFAULT
SIDES = ("A", "B")
SIDE_NAME = {"A": "side A   pounds " + chr(8594) + " dollars",
             "B": "side B   dollars " + chr(8594) + " pounds"}


def p_star(sigma, side):
    return closed_form_optimal_rate(a0, sigma, FEE, side=side)


def sigma_crit(side):
    """The volatility at which the optimal offer crosses the anchor."""
    return brentq(lambda s: p_star(s, side) - a0, 1e-4, SIGMA_HI)


if __name__ == "__main__":
    sig = np.linspace(SIGMA_LO, SIGMA_HI, N_SIGMA)
    sc = sigma_crit("A")           # the crossing is shared: both sides cross
                                   # at sqrt(2/pi)*fee*a0 (verified 16 Aug 26)
    print(f"v0 sigma sweep  |  a0 = {a0}, fee = {FEE:.0%}")
    print(f"  sigma_c = {sc:.4f}  (= sqrt(2/pi)*fee*a0 = "
          f"{np.sqrt(2 / np.pi) * FEE * a0:.4f}); the card's sigma = 0.05 "
          f"is in the fishing regime")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    P_by_side = {}
    for ax, side in zip(axes, SIDES):
        P = np.array([p_star(s, side) for s in sig])
        P_by_side[side] = P
        interior = np.array([offer_is_interior(Pi, a0, si, FEE, side)
                             for Pi, si in zip(P, sig)])
        if not interior.all():
            print(f"  side {side}: no interior optimum for sigma >= "
                  f"{sig[~interior].min():.4f} (offer channel dominated -- "
                  f"Bureau only); those points are left off the line")
        offer = np.where(interior, [trade_rate(Pi, side) for Pi in P], np.nan)
        anchor = trade_rate(a0, side)

        ax.plot(sig, offer, lw=2, color="#1a5276",
                label=f"optimal offer P*({chr(963)})")
        ax.axhline(anchor, ls="--", lw=1, color="0.55",
                   label=f"anchor = {anchor:.4f}")
        if side == "B" and SHOW_RECIPROCAL:
            # the naive symmetry, drawn so its failure is the visible point:
            # 1/P*_A and P*_B(GBP/$) touch only at sigma_c and split fast
            ax.plot(sig, 1.0 / P_by_side["A"], lw=1.2, ls=":",
                    color="#b03a2e", label="reciprocal of side A's optimum")
        ax.set_xlabel(r"rate volatility $\sigma$")
        ax.set_ylabel(f"optimal offer  [{rate_units(side)}]")
        # a shared x range, so where a line STOPS is visible rather than
        # hidden by autoscaling to the surviving points
        ax.set_xlim(SIGMA_LO, SIGMA_HI)
        ax.set_title(SIDE_NAME[side], fontsize=10)
        ax.legend(fontsize=8)

    card = 0.05
    print(f"  not reciprocal: at sigma = {card}, side B offers "
          f"{trade_rate(p_star(card, 'B'), 'B'):.4f} GBP/$ vs "
          f"1/P*_A = {1.0 / p_star(card, 'A'):.4f}; they coincide only "
          f"at sigma_c")

    fig.tight_layout()
    out = results_path(SAVE)
    fig.savefig(out, dpi=170)
    print(f"wrote {SAVE}")