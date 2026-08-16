"""
The rules below are the whole of the game's mechanics, and each is the
SAME rule seen from either end of a trade. `side` selects the end:
    "A"  the trader starts in pounds and wants dollars  
    "B"  the trader starts in dollars and wants pounds  
Every rule defaults to "A", so existing callers are unaffected.
"""
from dataclasses import dataclass

A0_DEFAULT = 1.25
sd_DEFAULT = 0.05
BDC_FEE_DEFAULT = 0.02


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
    #Target currency received per unit of initial currency spent, 
    #if an offer priced at P ($/GBP) is accepted.
    
    if side == "B":
        return 1.0 / P
    return P


def bdc_payoff_per_unit(X, f, side="A"):
    #The Bureau de Change's rate at revealed rate X, after 
    #its fee f, in target currency per unit of initial currency.
    if side == "B":
        return (1.0 - f) / X
    return (1.0 - f) * X

def rate_units(side="A"):
    #How to LABEL a target-per-unit-held rate for this side.

    return "$/GBP" if side == "A" else "GBP/$"

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results"

def results_path(filename: str) -> Path:
    """Absolute path into the Results folder, created on demand."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename