import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
================================================================================
 VERSION 0
 One liquidator, one round, one all-or-nothing offer, hidden true rate.
================================================================================

Structure of this file:
  0. THE TRADER        - TraderSpec: the one place to change who is playing
  1. GAME MODEL        - the objects that play out one round of the game
  2. CLOSED FORM       - the exact maths, solved with a bisection search
  3. MONTE CARLO       - simulate many games and average, as an independent
                         check on the closed form
  4. PLOT              - visualise expected payoff vs offer rate
================================================================================
"""

import math
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from Mechanics.fx_mechanics import (
    GameParams,
    mm_accepts,
    trade_rate,
    rate_units,
    bdc_payoff_per_unit,
    results_path,
    A0_DEFAULT,
    sd_DEFAULT,
    BDC_FEE_DEFAULT,
)

# The rules of the game, defined once and shared with v1. See fx_mechanics.py.


# ==============================================================================
# 0. THE TRADER
# ==============================================================================

@dataclass
class TraderSpec:

    side: str = "A"       # "A": starts in pounds, wants dollars (the T1 family)
                          # "B": starts in dollars, wants pounds (the T4 family)
    capital: float = 50_000.0
    params: GameParams = field(default_factory=GameParams)


def solve_v0(spec=None):
    #The whole of v0 for one spec: the best offer, and what it is worth.
    spec = spec or TraderSpec()
    p = spec.params
    quote = closed_form_optimal_rate(p.a0, p.sd, p.bdc_fee, spec.side)
    offer = trade_rate(quote, spec.side)
    value = expected_target_per_unit(quote, p.a0, p.sd, p.bdc_fee, spec.side)
    return quote, offer, value


# ==============================================================================
# 1. GAME MODEL
# ==============================================================================

class Trader:
    #Holds one currency, wants all of it in the other.
    #side "A" holds pounds and banks dollars, side "B" holds dollars and banks pounds.


    def __init__(self, held: float, banked: float = 0.0, side: str = "A"):
        self.held = held
        self.banked = banked
        self.side = side          # "A" sells pounds; "B" sells dollars

    def offer_everything_to_mm(self, offer_rate: float):
        """The trader's one offer: the whole book, at the proposed $/£ rate."""
        banked_requested = self.held * trade_rate(offer_rate, self.side)
        return self.held, banked_requested

    def receive_mm_trade(self, spent: float, received: float) -> None:
        self.held -= spent
        self.banked += received

    def use_bdc_for_everything(self, true_rate: float, fee: float) -> None:
        """Fallback if the MM rejects: convert the whole book at the BdC's fee.

        Uses the shared per-unit BdC payoff so the fee convention matches v1
        and every other version exactly."""
        self.banked += self.held * bdc_payoff_per_unit(true_rate, fee,
                                                       self.side)
        self.held = 0.0


class MarketMaker:
    def __init__(self, initial_ccy: float = 0.0, target_ccy: float = 0.0):
        # the MM's position in each of the trader's two currencies
        self.initial_ccy = initial_ccy
        self.target_ccy = target_ccy

    def accepts_trade(self, offer_rate: float, true_rate: float,
                      side: str = "A") -> bool:
        return mm_accepts(offer_rate, true_rate, side)

    def complete_trade(self, trader: Trader, spent: float, paid: float):
        trader.receive_mm_trade(spent, paid)
        self.initial_ccy += spent
        self.target_ccy -= paid


class BureauDeChange:
    def __init__(self, trader_fee: float = BDC_FEE_DEFAULT):
        self.trader_fee = trader_fee

    def exchange_for_trader(self, trader: Trader, true_rate: float):
        trader.use_bdc_for_everything(true_rate=true_rate, fee=self.trader_fee)


class GameResult:
    def __init__(self, true_rate: float, accepted_by_mm: bool,
                 final_held: float, final_banked: float):
        self.true_rate = true_rate
        self.accepted_by_mm = accepted_by_mm
        self.final_held = final_held
        self.final_banked = final_banked


class OneRoundGame:
    def __init__(
        self,
        initial_rate: float = A0_DEFAULT,
        sd: float = sd_DEFAULT,
        initial_capital: float = 50000,
        bdc_fee: float = BDC_FEE_DEFAULT,
        side: str = "A",
    ):
        self.initial_rate = initial_rate
        self.sd = sd
        self.initial_capital = initial_capital
        self.bdc_fee = bdc_fee
        self.side = side

    def play(self, offer_rate: float):
        trader = Trader(held=self.initial_capital, side=self.side)
        market_maker = MarketMaker()
        bdc = BureauDeChange(self.bdc_fee)

        true_rate = np.random.normal(self.initial_rate, self.sd)

        spent, requested = trader.offer_everything_to_mm(offer_rate)

        accepted = market_maker.accepts_trade(offer_rate, true_rate, self.side)

        if accepted:
            market_maker.complete_trade(trader, spent, requested)
        else:
            bdc.exchange_for_trader(trader, true_rate)

        return GameResult(
            true_rate=true_rate,
            accepted_by_mm=accepted,
            final_held=trader.held,
            final_banked=trader.banked,
        )


# ==============================================================================
# 2. CLOSED FORM
# ==============================================================================
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad


def h(offer_rate, a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT, side="A"):
    # The first-order condition, and the two sides differ in ONE term: the
    # probability the offer is taken. The seller is taken when the price is
    # below the true rate (1 - Phi), the buyer when it is above (Phi). The
    # marginal cost of shading, f * (P/sd) * phi, is identical.
    z = (offer_rate - a0) / sd
    if side == "B":
        return norm.cdf(z) - f * (offer_rate / sd) * norm.pdf(z)
    return (1 - norm.cdf(z)) - f * (offer_rate / sd) * norm.pdf(z)


def closed_form_optimal_rate(
    a0=A0_DEFAULT,
    sd=sd_DEFAULT,
    f=BDC_FEE_DEFAULT,
    side="A",
):
    # Return the optimal quote within the positive ±8σ range.
    if not 0.0 <= f <= 1.0:
        raise ValueError("bdc_fee must be between 0 and 1")

    lo = max(1e-9, a0 - 8.0 * sd)
    hi = a0 + 8.0 * sd
    fn = lambda P: h(P, a0, sd, f, side)

    hlo, hhi = fn(lo), fn(hi)

    # Interior optimum
    if hlo * hhi < 0.0:
        return brentq(fn, lo, hi)

    # With very small/zero fees, there may be no finite root.
    # In that case the optimum is one of the no-trade boundaries.
    return max(
        (lo, hi),
        key=lambda P: expected_target_per_unit(P, a0, sd, f, side),
    )


def expected_target_per_unit(P, a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT,
                               side="A"):
    """g(P) = P*(1-Phi(z)) + (1-f)*(a0*Phi(z) - sd*phi(z)),  z=(P-a0)/sd

    The buyer's counterpart is NOT a closed form. It receives 1/P per dollar if
    taken (probability Phi(z)) and (1-f)/X at the BdC otherwise, and
    E[1/X ; X > P] has no elementary antiderivative -- so that branch is one
    adaptive quadrature."""
    z = (P - a0) / sd
    if side == "B":
        tail, _ = quad(lambda x: norm.pdf(x, a0, sd) / x, P, a0 + 12 * sd,
                       epsabs=1e-14, epsrel=1e-14, limit=200)
        return norm.cdf(z) / P + (1 - f) * tail
    return P * (1 - norm.cdf(z)) + (1 - f) * (a0 * norm.cdf(z) - sd * norm.pdf(z))


def perfect_information_value(a0=A0_DEFAULT, sd=sd_DEFAULT, side="A"):
    """What one unit of initial currency is worth if the rate were KNOWN.

    The seller would sell at X and earn E[X] = a0; the buyer would buy at X and
    earn E[1/X].
    """
    if side == "B":
        val, _ = quad(lambda x: norm.pdf(x, a0, sd) / x,
                      a0 - 12 * sd, a0 + 12 * sd,
                      epsabs=1e-14, epsrel=1e-14, limit=200)
        return val
    return a0


# ==============================================================================
# 3. MONTE CARLO  (independent check via simulation)
# ==============================================================================

def monte_carlo_expected_target_per_unit(offer_rate: float, a0: float, sd: float, fee: float,
                                            n_games: int = 200_000, side: str = "A"):

    game = OneRoundGame(initial_rate=a0, sd=sd, initial_capital=1.0, bdc_fee=fee,
                        side=side)

    payoffs = [game.play(offer_rate).final_banked for _ in range(n_games)]
    mean = sum(payoffs) / n_games
    se = float(np.std(payoffs)) / math.sqrt(n_games)
    return mean, se

# ==============================================================================
# 4. PLOT
# ==============================================================================

def plot(spec=None, n_games: int = 5_000, save_path: str = "v0_plot.png"):

    spec = spec or TraderSpec()
    a0, sd, fee, side = spec.params.a0, spec.params.sd, spec.params.bdc_fee, spec.side

    # the seller bids UP from the anchor, the buyer its bid DOWN
    rates = (np.arange(a0 - 4 * sd, a0 + 2 * sd, 0.0025) if side == "B"
             else np.arange(a0 - 2 * sd, a0 + 4 * sd, 0.0025))

    mc_curve = [monte_carlo_expected_target_per_unit(P, a0, sd, fee, n_games,
                                                     side)[0]
                for P in rates]
    cf_curve = [expected_target_per_unit(P, a0, sd, fee, side) for P in rates]
    P_star = closed_form_optimal_rate(a0, sd, fee, side)
    bdc_only = (1 - fee) * perfect_information_value(a0, sd, side)

    # show the offer axis in the TRADER's units: for the buyer that reverses
    # the direction, so both sides read the same way -- left is cautious,
    # right is greedy
    unit = rate_units(side)
    x = [trade_rate(P, side) for P in rates]
    if side == "B":
        x, mc_curve, cf_curve = x[::-1], mc_curve[::-1], cf_curve[::-1]

    plt.figure(figsize=(8, 5))
    plt.plot(x, mc_curve, lw=1, alpha=0.55, label=f"Monte Carlo ({n_games:,}/pt)")
    plt.plot(x, cf_curve, lw=2, label="Closed form g(P)")
    plt.axvline(trade_rate(P_star, side), ls="--", color="k",
                label=f"offer* = {trade_rate(P_star, side):.4f} {unit}"
                      f"  (slip quote {P_star:.4f} $/GBP, {(P_star-a0)/sd:+.2f}σ)")
    plt.axhline(bdc_only, ls=":", color="grey",
                label=f"BdC-only = {bdc_only:.4f}")
    plt.xlabel(f"Offer rate ({unit} -- target currency per unit held)")
    plt.ylabel(f"Expected target per unit held ({unit})")
    plt.title(f"Version 0 (side {side}): expected payoff vs offer rate")
    plt.legend()
    plt.tight_layout()
    out = results_path(save_path)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved plot to {out}")

if __name__ == "__main__":
    spec = TraderSpec()
    quote, offer, g = solve_v0(spec)
    p = spec.params
    pinf = perfect_information_value(p.a0, p.sd, spec.side)
    floor = (1 - p.bdc_fee) * pinf
    unit = rate_units(spec.side)
    print(f"v0, side {spec.side}  (holds "
          f"{'pounds, wants dollars' if spec.side == 'A' else 'dollars, wants pounds'})")
    print(f"  best offer                {offer:.4f} {unit}")
    print(f"  expected target per unit  {g:.6f} {unit}")
    print(f"  BdC-only floor            {floor:.6f} {unit}")
    print(f"  perfect information       {pinf:.6f} {unit}")
    print(f"  share of the fee recovered "
          f"{100 * (g - floor) / (pinf - floor):.1f}%")
    plot(spec)