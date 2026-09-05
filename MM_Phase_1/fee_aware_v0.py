import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
================================================================================
 VERSION 0  (fee-aware)
 One liquidator, one round, one all-or-nothing offer, hidden true rate.
================================================================================

 The market maker's fee, in one paragraph. This twin adds g, the MM's OWN
 Bureau cost when it clears its book: never charged to the trader, never on
 the slip, but it moves where the MM breaks even. The seller is taken iff
 P < (1-g) X, the buyer iff P > X / (1-g). The optimisation variable is
 therefore the acceptance THRESHOLD X*, not the price -- the fill event is
 X > X* for the seller and X < X* for the buyer, exactly v0's geometry -- and
 the price on the slip is the quote, (1-g) X* for the seller, X* / (1-g) for
 the buyer. At g = 0 threshold and quote coincide and every number is v0's.

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


def mm_accepts_fee(P, X, side="A", g=0.0):
    """fx_mechanics.mm_accepts with the MM's own Bureau cost g: the MM only
    trades when the trade is good for the MM AFTER clearing its book."""
    if side == "B":
        return P > X / (1.0 - g)
    return P < (1.0 - g) * X


def quote_from_threshold(X, side="A", g=0.0):
    """The $/GBP number on the slip that implements acceptance threshold X:
    the seller must undercut it by the MM's fee, the buyer must overpay by it.
    At g = 0 this is the identity, and quote and threshold are the same
    number -- which is why v0 needs only one of them."""
    if side == "B":
        return X / (1.0 - g)
    return (1.0 - g) * X


# ==============================================================================
# 0. THE TRADER
# ==============================================================================

@dataclass
class TraderSpec:

    side: str = "A"       # "A": starts in pounds, wants dollars (the T1 family)
                          # "B": starts in dollars, wants pounds (the T4 family)
    capital: float = 50_000.0
    params: GameParams = field(default_factory=GameParams)
    mm_fee: float = 0.003   # the MM's own Bureau cost g; 0.0 is v0


def solve_v0(spec=None):
    #The whole of v0 for one spec: the best offer, and what it is worth.
    # The threshold is what is optimised; the quote is what goes on the slip.
    spec = spec or TraderSpec()
    p = spec.params
    threshold = closed_form_optimal_rate(p.a0, p.sd, p.bdc_fee, spec.side,
                                         spec.mm_fee)
    quote = quote_from_threshold(threshold, spec.side, spec.mm_fee)
    offer = trade_rate(quote, spec.side)
    value = expected_target_per_unit(threshold, p.a0, p.sd, p.bdc_fee,
                                     spec.side, spec.mm_fee)
    return threshold, quote, offer, value


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
    def __init__(self, initial_ccy: float = 0.0, target_ccy: float = 0.0,
                 mm_fee: float = 0.0):
        # the MM's position in each of the trader's two currencies
        self.initial_ccy = initial_ccy
        self.target_ccy = target_ccy
        self.mm_fee = mm_fee      # what clearing its own book costs it

    def accepts_trade(self, offer_rate: float, true_rate: float,
                      side: str = "A") -> bool:
        return mm_accepts_fee(offer_rate, true_rate, side, self.mm_fee)

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
        mm_fee: float = 0.0,
    ):
        self.initial_rate = initial_rate
        self.sd = sd
        self.initial_capital = initial_capital
        self.bdc_fee = bdc_fee
        self.side = side
        self.mm_fee = mm_fee

    def play(self, offer_rate: float):
        trader = Trader(held=self.initial_capital, side=self.side)
        market_maker = MarketMaker(mm_fee=self.mm_fee)
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


def h(threshold, a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT, side="A",
      g=0.0):
    # The first-order condition, and the two sides differ in ONE term: the
    # probability the offer is taken. The seller is taken when the threshold is
    # below the true rate (1 - Phi), the buyer when it is above (Phi). The
    # marginal cost of shading, f * (X/sd) * phi, is identical.
    #
    # The MM's fee enters that cost term and nowhere else. Differentiating the
    # fee-aware payoff, the fill branch carries (1-g) and the Bureau branch
    # (1-f), so dividing through by (1-g) leaves v0's condition with the fee
    # (f - g) / (1 - g): the fraction of the spread the trader still owns.
    z = (threshold - a0) / sd
    fee = (f - g) / (1.0 - g)
    if side == "B":
        return norm.cdf(z) - fee * (threshold / sd) * norm.pdf(z)
    return (1 - norm.cdf(z)) - fee * (threshold / sd) * norm.pdf(z)


def closed_form_optimal_rate(
    a0=A0_DEFAULT,
    sd=sd_DEFAULT,
    f=BDC_FEE_DEFAULT,
    side="A",
    g=0.0,
):
    # Return the optimal THRESHOLD within the positive ±8σ range. The quote
    # that implements it is quote_from_threshold(...); at g = 0 they are equal.
    if not 0.0 <= f <= 1.0:
        raise ValueError("bdc_fee must be between 0 and 1")
    if not 0.0 <= g < 1.0:
        raise ValueError("mm_fee must be at least 0 and below 1")

    lo = max(1e-9, a0 - 8.0 * sd)
    hi = a0 + 8.0 * sd
    fn = lambda X: h(X, a0, sd, f, side, g)

    hlo, hhi = fn(lo), fn(hi)

    # Interior optimum
    if hlo * hhi < 0.0:
        return brentq(fn, lo, hi)

    # With very small/zero fees, there may be no finite root -- and once the
    # MM's fee reaches the trader's, g >= f, the band (1-f) X < P < (1-g) X is
    # empty and no offer can beat the Bureau, so there is no root at ANY sigma.
    # In that case the optimum is one of the no-trade boundaries.
    return max(
        (lo, hi),
        key=lambda X: expected_target_per_unit(X, a0, sd, f, side, g),
    )


def offer_is_interior(X, a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT,
                      side="A", g=0.0):
    """Is X a genuine stationary point of g, or a search-bracket edge?

    closed_form_optimal_rate root-finds h inside a +/-8 sd bracket; when no
    interior root exists (side B once the offer channel is dominated, sigma
    >~ 0.088 at the card fee) it falls back to the better bracket EDGE, whose
    VALUE is the never-fill floor but whose location is just the bracket
    half-width -- numerics, not economics. The test must be RELATIVE: eight
    sd out, both FOC terms are ~1e-16, so their difference passes any
    absolute tolerance and a boundary impersonates a root. Scaled by the
    terms' own size, the ratio is 0 at a real optimum and O(1) at an edge.

    The MM's fee is in the cost term for the same reason it is in h: at g >= f
    the cost is zero or negative, no interior root exists, and this returns
    False at every sigma -- the offer channel is dominated outright."""
    cost = ((f - g) / (1.0 - g)) * (X / sd) * norm.pdf((X - a0) / sd)
    gap = h(X, a0, sd, f, side, g)               # fill - cost
    return abs(gap) / max(gap + 2.0 * cost, 1e-300) < 1e-6


def expected_target_per_unit(X, a0=A0_DEFAULT, sd=sd_DEFAULT, f=BDC_FEE_DEFAULT,
                               side="A", g=0.0):
    """g(X) = (1-g)*X*(1-Phi(z)) + (1-f)*(a0*Phi(z) - sd*phi(z)),  z=(X-a0)/sd

    X is the acceptance THRESHOLD. A fill pays the QUOTE, (1-g) X for the
    seller and (1-g)/X for the buyer; the Bureau branch never sees the MM's
    fee, which is the whole asymmetry the fee introduces.

    The buyer's counterpart is NOT a closed form. It receives (1-g)/X per
    dollar if taken (probability Phi(z)) and (1-f)/X at the BdC otherwise, and
    E[1/X ; X > X*] has no elementary antiderivative -- so that branch is one
    adaptive quadrature."""
    z = (X - a0) / sd
    if side == "B":
        tail, _ = quad(lambda x: norm.pdf(x, a0, sd) / x, X, a0 + 12 * sd,
                       epsabs=1e-14, epsrel=1e-14, limit=200)
        return (1 - g) * norm.cdf(z) / X + (1 - f) * tail
    return ((1 - g) * X * (1 - norm.cdf(z))
            + (1 - f) * (a0 * norm.cdf(z) - sd * norm.pdf(z)))


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
                                            n_games: int = 200_000, side: str = "A",
                                            g: float = 0.0):
    # offer_rate is the QUOTE, the number the trader puts on the slip
    game = OneRoundGame(initial_rate=a0, sd=sd, initial_capital=1.0, bdc_fee=fee,
                        side=side, mm_fee=g)

    payoffs = [game.play(offer_rate).final_banked for _ in range(n_games)]
    mean = sum(payoffs) / n_games
    se = float(np.std(payoffs)) / math.sqrt(n_games)
    return mean, se

# ==============================================================================
# 4. PLOT
# ==============================================================================

def plot(spec=None, n_games: int = 5_000, save_path: str = "v0_fee_aware_plot.png"):

    spec = spec or TraderSpec()
    a0, sd, fee, side = spec.params.a0, spec.params.sd, spec.params.bdc_fee, spec.side
    g = spec.mm_fee

    # the seller bids UP from the anchor, the buyer its bid DOWN. These are
    # THRESHOLDS; each is plotted at the quote that implements it, so the axis
    # stays what it says it is -- the rate the trader writes on the slip.
    rates = (np.arange(a0 - 4 * sd, a0 + 2 * sd, 0.0025) if side == "B"
             else np.arange(a0 - 2 * sd, a0 + 4 * sd, 0.0025))
    quotes = [quote_from_threshold(X, side, g) for X in rates]

    mc_curve = [monte_carlo_expected_target_per_unit(P, a0, sd, fee, n_games,
                                                     side, g)[0]
                for P in quotes]
    cf_curve = [expected_target_per_unit(X, a0, sd, fee, side, g) for X in rates]
    X_star = closed_form_optimal_rate(a0, sd, fee, side, g)
    P_star = quote_from_threshold(X_star, side, g)
    bdc_only = (1 - fee) * perfect_information_value(a0, sd, side)

    # show the offer axis in the TRADER's units: for the buyer that reverses
    # the direction, so both sides read the same way -- left is cautious,
    # right is greedy
    unit = rate_units(side)
    x = [trade_rate(P, side) for P in quotes]
    if side == "B":
        x, mc_curve, cf_curve = x[::-1], mc_curve[::-1], cf_curve[::-1]

    plt.figure(figsize=(8, 5))
    plt.plot(x, mc_curve, lw=1, alpha=0.55, label=f"Monte Carlo ({n_games:,}/pt)")
    plt.plot(x, cf_curve, lw=2, label="Closed form g(X)")
    if offer_is_interior(X_star, a0, sd, fee, side, g):
        plt.axvline(trade_rate(P_star, side), ls="--", color="k",
                    label=f"offer* = {trade_rate(P_star, side):.4f} {unit}"
                          f"  (slip quote {P_star:.4f} $/GBP, "
                          f"threshold {(X_star-a0)/sd:+.2f}σ)")
    else:
        # no interior optimum: the solver fell back to a bracket edge, an
        # arbitrary representative of the never-fill region. Its VALUE is the
        # BdC-only floor; its location is numerics, so draw no line for it.
        plt.plot([], [], " ",
                 label=f"no interior optimum at σ = {sd:g}, g = {g:g}: offer "
                       f"channel dominated -- never fill, BdC-only")
    plt.axhline(bdc_only, ls=":", color="grey",
                label=f"BdC-only = {bdc_only:.4f}")
    plt.xlabel(f"Offer rate ({unit} -- target currency per unit held)")
    plt.ylabel(f"Expected target per unit held ({unit})")
    plt.title(f"Version 0 (side {side}, MM fee g = {g:g}): "
              f"expected payoff vs offer rate")
    plt.legend()
    plt.tight_layout()
    out = results_path(save_path)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  saved plot to {out}")

if __name__ == "__main__":
    spec = TraderSpec()
    threshold, quote, offer, value = solve_v0(spec)
    p = spec.params
    pinf = perfect_information_value(p.a0, p.sd, spec.side)
    floor = (1 - p.bdc_fee) * pinf
    # even a clairvoyant trader pays a fee to whichever channel it uses, so
    # the attainable ceiling is the frictionless value less the cheaper of the
    # two. At g = 0 this is pinf and the share below is v0's.
    ceiling = (1 - min(spec.mm_fee, p.bdc_fee)) * pinf
    unit = rate_units(spec.side)
    print(f"v0, side {spec.side}, MM fee g = {spec.mm_fee:g}  (holds "
          f"{'pounds, wants dollars' if spec.side == 'A' else 'dollars, wants pounds'})")
    if offer_is_interior(threshold, p.a0, p.sd, p.bdc_fee, spec.side, spec.mm_fee):
        print(f"  best offer                {offer:.4f} {unit}"
              f"  (threshold {threshold:.4f} $/GBP, slip quote {quote:.4f})")
    else:
        print(f"  best offer                none -- offer channel dominated "
              f"(never fill; {offer:.4f} {unit} is the search-bracket edge, "
              f"not an optimum)")
    print(f"  expected target per unit  {value:.6f} {unit}")
    print(f"  BdC-only floor            {floor:.6f} {unit}")
    print(f"  attainable ceiling        {ceiling:.6f} {unit}")
    print(f"  perfect information       {pinf:.6f} {unit}")
    print(f"  share of the fee recovered "
          f"{100 * (value - floor) / (ceiling - floor):.1f}%")
    plot(spec)