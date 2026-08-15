"""
fx_mechanics -- RECONSTRUCTED from v2_multiple_rounds.py call sites.
The original module was not attached; interface inferred from usage:
    mm_accepts(P, X)            -> MM accepts a pound-sale iff P < X
    bdc_payoff_per_pound(X, f)  -> (1 - f) * X
    GameParams(a0, sd, bdc_fee) -> the shared parameters
!! DIFF THIS AGAINST THE REPOSITORY'S fx_mechanics.py BEFORE TRUSTING RESULTS.

The three rules below are the whole of the game's mechanics, and each is the
SAME rule seen from either end of a trade. `side` selects the end:
    "A"  the trader starts in pounds and wants dollars  (the T1 family)
    "B"  the trader starts in dollars and wants pounds  (the T4 family)
Every rule defaults to "A", so existing callers are unaffected.
"""
from dataclasses import dataclass

A0_DEFAULT = 1.25
sd_DEFAULT = 0.05
BDC_FEE_DEFAULT = 0.1


@dataclass
class GameParams:
    a0: float = A0_DEFAULT
    sd: float = sd_DEFAULT
    bdc_fee: float = BDC_FEE_DEFAULT


def mm_accepts(P, X, side="A"):
    """Does the MM take an offer priced at P ($/GBP) when the true rate is X?

    A trader SELLING pounds is taken iff its asking price undercuts the true
    rate; a trader BUYING pounds is taken iff its bid beats it. Both say the
    same thing: the MM only trades when the trade is good for the MM.
    """
    if side == "B":
        return P > X
    return P < X


def trade_rate(P, side="A"):
    """Target currency received per unit of initial currency spent, if an offer
    priced at P ($/GBP) is accepted.

    The seller hands over pounds and receives P dollars for each; the buyer
    hands over dollars and receives 1/P pounds for each. The reciprocal is the
    whole of the difference.
    """
    if side == "B":
        return 1.0 / P
    return P


def bdc_payoff_per_unit(X, f, side="A"):
    """The Bureau de Change's rate at revealed rate X, after its fee f, in
    target currency per unit of initial currency. (The name is the seller's;
    side "B" is the same rule for a trader converting the other way.)"""
    if side == "B":
        return (1.0 - f) / X
    return (1.0 - f) * X

def rate_units(side="A"):
    """How to LABEL a target-per-unit-held rate for this side.
 
    Prices on a trade slip are always quoted $/GBP, because that is the
    convention the hidden rate and mm_accepts are both stated in. But what a
    trader actually earns is target currency per unit it holds, and that is
    $/GBP only for the seller -- the buyer's is its reciprocal. Anything shown
    to a human should carry the trader's units, not the slip's.
    """
    return "$/GBP" if side == "A" else "GBP/$"