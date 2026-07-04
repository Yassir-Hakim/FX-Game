"""
================================================================================
 VERSION 0
 One liquidator, one round, one all-or-nothing offer, hidden true rate.
================================================================================

Structure of this file:
  1. GAME MODEL        - the objects that play out one round of the game
  2. CLOSED FORM       - the exact maths (derived by hand), solved with a
                         bisection search
  3. MONTE CARLO       - simulate many games and average, as an independent
                         check on the closed form
  4. PLOT              - visualise expected payoff vs offer rate

 The MM's accept/reject rule and the BdC fee are NOT defined in this file: they
 live in fx_mechanics.py and are imported, so v0 and v1 provably play the same
 game. See fx_mechanics.py for why.
================================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# The rules of the game, defined once and shared with v1. See fx_mechanics.py.
from fx_mechanics import (
    mm_accepts,
    bdc_payoff_per_pound,
    A0_DEFAULT,
    sd_DEFAULT,
    BDC_FEE_DEFAULT,
)

# ==============================================================================
# 1. GAME MODEL
# ==============================================================================

class Trader:
    """The liquidator: holds pounds, wants to convert all of them to dollars."""

    def __init__(self, pounds: float, dollars: float = 0.0):
        self.pounds = pounds
        self.dollars = dollars

    def sell_all_pounds_to_mm(self, offer_rate: float):
        """The trader's one offer: sell all pounds at the proposed $/£ rate."""
        dollars_requested = self.pounds * offer_rate
        return self.pounds, dollars_requested

    def receive_mm_trade(self, pounds_sold: float, dollars_received: float) -> None:
        self.pounds -= pounds_sold
        self.dollars += dollars_received

    def use_bdc_to_sell_all_pounds(self, true_rate: float, fee: float) -> None:
        """Fallback if the MM rejects: sell everything to the BdC at a fee.

        Uses the shared per-pound BdC payoff so the fee convention matches v1
        and every other version exactly."""
        self.dollars += self.pounds * bdc_payoff_per_pound(true_rate, fee)
        self.pounds = 0.0


class MarketMaker:
    def __init__(self, pounds: float = 0.0, dollars: float = 0.0):
        self.pounds = pounds
        self.dollars = dollars

    def accepts_trade(self, offer_rate: float, true_rate: float) -> bool:
        """Trader is selling pounds to the MM. Delegates to the shared rule so
        v0 and v1 can never disagree about when a trade is accepted."""
        return mm_accepts(offer_rate, true_rate)

    def complete_trade(self, trader: Trader, pounds_sold: float, dollars_paid: float):
        trader.receive_mm_trade(pounds_sold, dollars_paid)
        self.pounds += pounds_sold
        self.dollars -= dollars_paid


class BureauDeChange:
    def __init__(self, trader_fee: float = BDC_FEE_DEFAULT):
        self.trader_fee = trader_fee

    def exchange_for_trader(self, trader: Trader, true_rate: float):
        trader.use_bdc_to_sell_all_pounds(true_rate=true_rate, fee=self.trader_fee)


class GameResult:
    def __init__(self, true_rate: float, accepted_by_mm: bool, final_pounds: float, final_dollars: float):
        self.true_rate = true_rate
        self.accepted_by_mm = accepted_by_mm
        self.final_pounds = final_pounds
        self.final_dollars = final_dollars


class OneRoundGame:
    def __init__(
        self,
        initial_rate: float = A0_DEFAULT,
        sd: float = sd_DEFAULT,
        initial_pounds: float = 50000,
        bdc_fee: float = BDC_FEE_DEFAULT,
    ):
        self.initial_rate = initial_rate
        self.sd = sd
        self.initial_pounds = initial_pounds
        self.bdc_fee = bdc_fee

    def play(self, offer_rate: float):
        trader = Trader(pounds=self.initial_pounds)
        market_maker = MarketMaker()
        bdc = BureauDeChange(self.bdc_fee)

        true_rate = np.random.normal(self.initial_rate, self.sd)

        pounds_sold, dollars_requested = trader.sell_all_pounds_to_mm(offer_rate)

        accepted = market_maker.accepts_trade(offer_rate, true_rate)

        if accepted:
            market_maker.complete_trade(trader, pounds_sold, dollars_requested)
        else:
            bdc.exchange_for_trader(trader, true_rate)

        return GameResult(
            true_rate=true_rate,
            accepted_by_mm=accepted,
            final_pounds=trader.pounds,
            final_dollars=trader.dollars,
        )


# ==============================================================================
# 2. CLOSED FORM
# ==============================================================================
from scipy.stats import norm
from scipy.optimize import brentq


def h(offer_rate, a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT):
    z = (offer_rate - a0) / sd
    return (1 - norm.cdf(z)) - f * (offer_rate / sd) * norm.pdf(z)


def closed_form_optimal_rate(a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT):
    """Solves h(P)=0 for P* via bisection (brentq), searching between a0 and a0+8σ."""
    return brentq(lambda P: h(P, a0, sd, f), a0, a0 + 8 * sd)


def expected_dollars_per_pound(P, a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT):
    """g(P) = P*(1-Phi(z)) + (1-f)*(a0*Phi(z) - sd*phi(z)),  z=(P-a0)/sd"""
    z = (P - a0) / sd
    return P * (1 - norm.cdf(z)) + (1 - f) * (a0 * norm.cdf(z) - sd * norm.pdf(z))


# ==============================================================================
# 3. MONTE CARLO  (independent check via simulation)
# ==============================================================================

def estimate_average_result(offer_rate: float, n_games: int, game: OneRoundGame = None):
    game = game or OneRoundGame()

    total_dollars = 0.0

    for _ in range(n_games):
        result = game.play(offer_rate)
        total_dollars += result.final_dollars

    return total_dollars / n_games


def monte_carlo_expected_dollars_per_pound(offer_rate: float, a0: float, sd: float, fee: float,
                                            n_games: int = 200_000):
    """
    Wraps estimate_average_result to report a per-pound value with a standard
    error, so it can be compared apples-to-apples with the closed form.
    Uses initial_pounds=1.0 so "total dollars" IS "dollars per pound".
    """
    game = OneRoundGame(initial_rate=a0, sd=sd, initial_pounds=1.0, bdc_fee=fee)

    payoffs = [game.play(offer_rate).final_dollars for _ in range(n_games)]
    mean = sum(payoffs) / n_games
    se = float(np.std(payoffs)) / math.sqrt(n_games)
    return mean, se

# ==============================================================================
# 4. PLOT
# ==============================================================================

def plot(a0: float = A0_DEFAULT, sd: float = sd_DEFAULT, fee: float = BDC_FEE_DEFAULT,
         n_games: int = 5_000, save_path: str = "v0_plot.png"):

    rates = np.arange(a0 - 2 * sd, a0 + 4 * sd, 0.0025)

    mc_curve = [monte_carlo_expected_dollars_per_pound(P, a0, sd, fee, n_games)[0]
                for P in rates]
    cf_curve = [expected_dollars_per_pound(P, a0, sd, fee) for P in rates]
    P_star = closed_form_optimal_rate(a0, sd, fee)

    plt.figure(figsize=(8, 5))
    plt.plot(rates, mc_curve, lw=1, alpha=0.55, label=f"Monte Carlo ({n_games:,}/pt)")
    plt.plot(rates, cf_curve, lw=2, label="Closed form g(P)")
    plt.axvline(P_star, ls="--", color="k",
                label=f"P* = {P_star:.4f}  (a0 + {(P_star-a0)/sd:.2f}σ)")
    plt.axhline((1 - fee) * a0, ls=":", color="grey",
                label=f"BdC-only = {(1-fee)*a0:.4f}")
    plt.xlabel("Offer rate P ($ per £)")
    plt.ylabel("Expected dollars per pound")
    plt.title("Version 0: expected payoff vs offer rate")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot()
