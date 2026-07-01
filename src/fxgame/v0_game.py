"""
================================================================================
 VERSION 0
 One liquidator, one round, one all-or-nothing offer, hidden true rate.
================================================================================

Structure of this file:
  1. GAME MODEL       - the objects that play out one round of the game
  2. CLOSED FORM       - the exact maths (derived by hand), solved with a
                         bisection search
  3. MONTE CARLO       - simulate many games and average, as an independent
                         check on the closed form
  4. PLOT              - visualise expected payoff vs offer rate
================================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt

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
        """Fallback if the MM rejects: sell everything to the BdC at a fee."""
        self.dollars += self.pounds * (1 - fee) * true_rate
        self.pounds = 0.0


class MarketMaker:
    def __init__(self, pounds: float = 0.0, dollars: float = 0.0):
        self.pounds = pounds
        self.dollars = dollars

    def accepts_trade(self, offer_rate: float, true_rate: float) -> bool:
        """
        Trader is selling pounds to the MM.

        MM accepts if the trader's offered rate is below the true rate.
        Example:
        true rate = 1.30
        trader offers = 1.27
        MM buys pounds cheaply, so accepts.
        """
        return offer_rate < true_rate
    
    def complete_trade(self, trader: Trader, pounds_sold: float, dollars_paid: float):
        trader.receive_mm_trade(pounds_sold, dollars_paid)
        self.pounds += pounds_sold
        self.dollars -= dollars_paid


class BureauDeChange:
    def __init__(self, trader_fee: float = 0.02):
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
        initial_rate: float = 1.25,
        sigma: float = 0.05,
        initial_pounds: float = 50000,
        bdc_fee: float = 0.02,          
    ):
        self.initial_rate = initial_rate
        self.sigma = sigma
        self.initial_pounds = initial_pounds
        self.bdc_fee = bdc_fee          

    def play(self, offer_rate: float):
        trader = Trader(pounds=self.initial_pounds)
        market_maker = MarketMaker()
        bdc = BureauDeChange(self.bdc_fee)   

        true_rate = np.random.normal(self.initial_rate, self.sigma)

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


def h(offer_rate, a0=1.25, sigma=0.05, f=0.02):
    z = (offer_rate - a0) / sigma
    return (1 - norm.cdf(z)) - f * (offer_rate / sigma) * norm.pdf(z)


def closed_form_optimal_rate(a0=1.25, sigma=0.05, f=0.02):
    """Solves h(P)=0 for P* via bisection (brentq), searching between a0 and a0+8σ."""
    return brentq(lambda P: h(P, a0, sigma, f), a0, a0 + 8 * sigma)


def expected_dollars_per_pound(P, a0=1.25, sigma=0.05, f=0.02):
    """g(P) = P*(1-Phi(z)) + (1-f)*(a0*Phi(z) - sigma*phi(z)),  z=(P-a0)/sigma"""
    z = (P - a0) / sigma
    return P * (1 - norm.cdf(z)) + (1 - f) * (a0 * norm.cdf(z) - sigma * norm.pdf(z))


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


def monte_carlo_expected_dollars_per_pound(offer_rate: float, a0: float, sigma: float, fee: float,
                                            n_games: int = 200_000):
    """
    Wraps estimate_average_result to report a per-pound value with a standard
    error, so it can be compared apples-to-apples with the closed form.
    Uses initial_pounds=1.0 so "total dollars" IS "dollars per pound".
    """
    game = OneRoundGame(initial_rate=a0, sigma=sigma, initial_pounds=1.0, bdc_fee=fee)

    # one honest pass through your loop, collecting every outcome
    payoffs = [game.play(offer_rate).final_dollars for _ in range(n_games)]
    mean = sum(payoffs) / n_games                       # = estimate_average_result(offer_rate, n_games, game)
    se = float(np.std(payoffs)) / math.sqrt(n_games)
    return mean, se

# ==============================================================================
# 4. PLOT
# ==============================================================================

def plot(a0: float = 1.25, sigma: float = 0.05, fee: float = 0.02,
         n_games: int = 5_000, save_path: str = "v0_plot.png"):

    rates = np.arange(a0 - 2 * sigma, a0 + 4 * sigma, 0.0025)

    mc_curve = [monte_carlo_expected_dollars_per_pound(P, a0, sigma, fee, n_games)[0]
                for P in rates]
    cf_curve = [expected_dollars_per_pound(P, a0, sigma, fee) for P in rates]
    P_star = closed_form_optimal_rate(a0, sigma, fee)

    plt.figure(figsize=(8, 5))
    plt.plot(rates, mc_curve, lw=1, alpha=0.55, label=f"Monte Carlo ({n_games:,}/pt)")
    plt.plot(rates, cf_curve, lw=2, label="Closed form g(P)")
    plt.axvline(P_star, ls="--", color="k",
                label=f"P* = {P_star:.4f}  (a0 + {(P_star-a0)/sigma:.2f}σ)")
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