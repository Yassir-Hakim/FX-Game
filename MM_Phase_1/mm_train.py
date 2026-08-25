import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "NN_one_offer_game"      # the folder holding rl_env.py and
                                         # torch_train.py; edit this ONE name
                                         # if that folder is ever renamed
assert (_CORE / "rl_env.py").exists(), f"core folder not found: {_CORE}"
sys.path.insert(0, str(_CORE))           # the certified machinery layer
sys.path.insert(0, str(_ROOT))           # Mechanics/, DP_Models/
"""
  mm_train.py -- a LEARNED market maker against the FROZEN traders.
  Direct policy optimisation through a differentiable simulator, the same
  neural stochastic control as torch_train, with the MM's end-of-game P/L
  as the objective and the trader frozen inside the loop. Side A is trained
  and reported first, then side B, in one run.

THE MM'S GAME (rules doc, MM sheet)
  No initial capital. Sees the true rate every round and each offer as it
  comes; accepts or rejects. Carries a book of pounds and dollars between
  rounds, either leg may go negative. At the end of every round it may
  repay part of its short leg at the Bureau, fee MM_FEE on the currency it
  sells, never more than its long leg can fund (see mm_bureau); at round 5
  it MUST clear any short at the round-5 rate, same fee.
  What is left is converted into pounds at the round-5 rate with no fee and
  scored as a fraction of the game's initial capital in pounds at a0.

THE MM'S ACTION: a private break-even level, a multiple of the true rate,
  set BEFORE the offer's price is read (a market maker's reservation
  price). An offer is accepted iff it clears that level on the favourable
  side. Avi's hypothesis is then a number: at fee 0 the level should come
  out at the true rate itself.
    THE LEVEL IS THE MM'S PRICE. In torch_train the trader quotes a price
  and the offer is measured against the HIDDEN RATE; here the MM quotes a
  level and the offer is measured against THAT. Same comparison, same jump
  in the payoff, so the same annealed ramp -- tau -- makes it learnable.
  A free logit head ("accept probability") was tried instead and collapsed:
  it falls into "reject all", where the sigmoid has no gradient, before it
  can tell offers apart. A level in price units cannot do that: it just
  moves through the offers.
    The level's sigmoid has its midpoint at exactly 1.0 x the true rate,
  which IS the fee-0 answer, so the parameterisation could be accused of
  handing it over. It does not: rerun mm_squash with 1.5 in place of 2.0
  (midpoint 0.75 x X, a 25% displacement) and side A still lands on 0.9992
  with 99.8% of volume matching the naive rule and the same P/L to 0.1 bp.

WHAT THE MM SEES: rounds remaining, the true rate, its own book. NOT the
  offer's price (it goes into the verdict, not the level), NOT the trader's
  book, card or history, NOT the rate path, NOT any distribution parameter.

MEASUREMENT (project standard, not learner features)
  Three checks abort the run unless this file's game is the certified one:
  the trader's P/L under the naive MM must equal torch_train.rollout to
  machine precision (check 1); every fill must move the trader's and the
  MM's books by equal and opposite amounts and every Bureau trade must move
  value at exactly the fee-adjusted rate (check 2); the MM's P/L under the
  hardcoded rules must match closed forms (check 3). All reported numbers
  are hard-game replays paired with the hardcoded rules on the same paths.
  The learner is told nothing by any of this.

DISTRIBUTION KNOWLEDGE: none. No setting encodes a0 or sd. The MM's one
  constant is the game's published capital at the round-0 rate.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

import NN_one_offer_game.torch_train as T
from Mechanics.fx_mechanics import mm_accepts, results_path
from train_traders import CARD, _trader_fingerprint

# ============================ SETTINGS ======================================
SIDES = ("A", "B")      # side A first: its validation is exact (see main)
MM_FEE = 0.01           # the MM's Bureau fee. 0.0 for the validation run
                        # (learner should flatten on the naive rule), 0.01
                        # for the game as written. 
GATE = "smoothed"       # "smoothed": annealed sigmoid ramp on the verdict
                        # "hard":     hard verdicts for training. For the
                        #             trader this is a biased gradient; for
                        #             the MM it is NO gradient 
TAU_START = 0.2         #begin with a ramp 0.2 rate-move wide
TAU_END = 0.02          # end at one-tenth of that, essentially the true rule,
                        # so the level is placed for the REAL verdict and not
                        # for the smoothed one, to ~1e-3 of the rate.
TAU_ANNEAL_FRAC = 0.35  # anneal tau over the FIRST this-fraction of training,
                        # then HOLD at TAU_END. torch_train's reason applies
                        # unchanged: without it the schedule stretches with
                        # ITERS, so a longer run spends longer in the soft
                        # game. Evaluation is always the hard rule.
LEVEL_INIT_OFFSET = 0.1 # start the level 10% off the true rate on the
                        # CONSERVATIVE side (rejects everything, P/L 0), so
                        # the curve shows the rule being FOUND, not assumed
                        # by the initialisation. Conservative, not generous:
                        # from the generous side every offer says "lower
                        # it", the level sprints through the offers and
                        # stops dead below them, out of the ramp's reach
                        # (measured). From below it climbs into the
                        # favourable offers and the sign flips at the rule.
SEE_OFFER_SIZE = False  # also show the MM the offered amount. Off: with one
                        # trader the per-unit verdict does not depend on it,
                        # and it is a window into the trader's book
ITERS = 2000            # training iterations, one fresh Monte Carlo batch each
BATCH = 1000            # games per iteration
LR = 1e-4               # AdamW, constant.
CLIP_NORM = 1.0         # cap on how far one step can move the weights, so a
                        # freak batch can't wreck the policy
VAL_EVERY = 50          # hard-game validation cadence; best checkpoint kept
VAL_PATHS = 20_000
EVAL_PATHS = 200_000    # final hard-game evaluation, paired with the rules
RECORD_PATHS = 20_000   # fresh paths replayed WITH a record of the play, for
                        # mm_diagnostics (the level and boundary plots);
                        # kept small so the npz stays small
CHECKS = True           # checks 1-3 must pass before any training
SEED_INIT = 42          # MM network init (fork_rng, as torch_train)
SEED_PATH = 5678        # MM path noise; not the trader's SEED_PATH
# ============================================================================

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# 1. models -- the trader's own network class, re-used for the MM
# ---------------------------------------------------------------------------
def mm_squash(raw, anchor):
    #Turn the head's two unbounded numbers into the MM's two decisions
    break_even = 2.0 * anchor * torch.sigmoid(raw[..., 0])  #factor 2 makes levels above the anchor reachable
    repay_fraction = torch.sigmoid(raw[..., 1])             #repaid at the Bureau
    return break_even, repay_fraction


def build_mm_net(seed, side):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = T.AlphaNet(n_in=N_MM_FEAT, n_out=2)
    fill_direction = 1.0 if side == "A" else -1.0
    level = 1.0 - fill_direction * LEVEL_INIT_OFFSET #sets so the learning curve starts at zero
    with torch.no_grad():
        net.head.weight[0].zero_()
        net.head.bias[0] = torch.logit(torch.tensor(level / 2.0))
    return net


def load_trader(side):
    """The FROZEN trader for this side: train_traders.py's card and its
    policy_best.pth, refused unless the DONE fingerprint matches the card and
    torch_train's settings as they are NOW. Weights frozen; the graph through
    its forward pass stays live, so the MM's gradient sees how a verdict
    changes the trader's later offers."""
    spec = CARD[side]
    run = results_path(f"mm_phase1/trader_{side}")
    done_marker = run / "DONE"
    assert done_marker.exists(), (
        f"no frozen trader for side {side}: run train_traders.py first")
    fingerprint = _trader_fingerprint(spec)
    assert done_marker.read_text().strip() == fingerprint, (
        f"side {side}: the frozen trader is STALE (its DONE fingerprint does "
        f"not match the current card or torch_train settings); re-run "
        f"train_traders.py")
    net = T.build_net(T.SEED_INIT)
    net.load_state_dict(torch.load(run / "policy_best.pth"))
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return spec, net, fingerprint


# ---------------------------------------------------------------------------
# 2. the game with a live market maker, explicit in torch ops
# ---------------------------------------------------------------------------
N_MM_FEAT = 5 + int(SEE_OFFER_SIZE)   

def game_capital_pounds(spec):
    #The game's total initial capital in pounds at the published round-0
    #rate: the MM's scoring unit (first row of its accounting sheet)
    return spec.L if spec.side == "A" else spec.L / spec.params.a0


# The five numbers the market maker can see, assembled into one row per game.
def mm_features(rounds_remaining_fraction, phase_flag, hidden_rate_now,
                mm_pounds, mm_dollars, offered_amount, spec):

    n_games = hidden_rate_now.shape[0]
    capital = game_capital_pounds(spec)
    a0 = spec.params.a0

    def as_column(value):
        return (torch.full((n_games,), float(value))
                if np.isscalar(value) else value)

    feature_columns = [
        as_column(rounds_remaining_fraction),  # 1  in [0,1], NOT the raw round
        as_column(phase_flag),                 # 2  0 = verdict, 1 = at the Bureau
        hidden_rate_now,                       # 3  the true rate, MM's private info
        mm_pounds / capital,                   # 4  the book as fractions of the
        mm_dollars / (a0 * capital),           # 5  game's capital; negative = short
    ]
    if SEE_OFFER_SIZE:
        offer_unit = capital if spec.side == "A" else a0 * capital
        feature_columns.append(offered_amount / offer_unit)   # 6  offer size
    return torch.stack(feature_columns, dim=-1)


def mm_bureau(mm_pounds, mm_dollars, repay_fraction, rate, fee):
    #calculates the MM's negative book
    short_dollars = torch.clamp(-mm_dollars, min=0.0)
    short_pounds = torch.clamp(-mm_pounds, min=0.0)
    #min of how much you want and how much you can afford
    bought_dollars = torch.minimum(
        repay_fraction * short_dollars,
        torch.clamp(mm_pounds, min=0.0) * (1.0 - fee) * rate)
    bought_pounds = torch.minimum(
        repay_fraction * short_pounds,
        torch.clamp(mm_dollars, min=0.0) * (1.0 - fee) / rate)
    #recluclationg book after trade
    mm_pounds = mm_pounds + bought_pounds - bought_dollars / ((1.0 - fee) * rate)
    mm_dollars = mm_dollars + bought_dollars - bought_pounds * rate / (1.0 - fee)
    return mm_pounds, mm_dollars

#4 hardcoded MM's based of being fee aware or not, and choosing to repay at that round or the final
@dataclass
class Rule:
    """A hardcoded MM."""
    name: str
    fee_aware: bool      # break-even moved by the MM's own fee; False: at X
    repay: float         # Bureau repay fraction every round: 1.0 flatten - clear negative that round,
                         # 0.0 carry the short to round 5


def rule_verdict(rule, quoted_price, hidden_rate_now, side, fee):
    #1.0 accept / 0.0 reject, elementwise, plus the level the rule operates,
    #so a rule replay's record is truthful. fx_mechanics.mm_accepts at the
    #true rate, or at the rate shifted so a fill breaks even after the MM's
    #own Bureau fee: P < (1-g)X for a pound seller, P > X/(1-g) for a buyer
    break_even = hidden_rate_now
    if rule.fee_aware:
        break_even = ((1.0 - fee) * hidden_rate_now if side == "A"
                      else hidden_rate_now / (1.0 - fee))
    return mm_accepts(quoted_price, break_even, side).double(), break_even


def reference_rules(fee):
    #The hardcoded MMs the learner is measured against. At fee 0 the fee-aware
    #rules ARE the naive ones, so only the naive pair is reported.
    rules = [Rule("naive+flatten", False, 1.0), Rule("naive+carry", False, 0.0)]
    if fee > 0.0:
        rules += [Rule("fee-aware+flatten", True, 1.0),
                  Rule("fee-aware+carry", True, 0.0)]
    return rules


RECORD_KEYS = ("quoted_price", "offered_amount", "hidden_rate", "break_even",
               "fill_weight", "repay_fraction",
               "trader_held_after_fill", "trader_banked_after_fill",
               "trader_held_after_bureau", "trader_banked_after_bureau",
               "mm_pounds_after_fill", "mm_dollars_after_fill",
               "mm_pounds_after_bureau", "mm_dollars_after_bureau")


# The game itself: the frozen trader quoting, the MM judging, both books
# kept, re-written in torch so autodiff can see through it. Vectorised;
# returns one MM P/L and one trader P/L per game.
def rollout(mm_net, trader_net, spec, rate_moves_sd, tau=None, rule=None,
            fee=None, record=False, tau_abs=False):
    """tau     None: hard verdicts, accept iff the offer clears the MM's level
                     -- THE REAL RULE, used for all evaluation. A number:
                     training, the ramp width in observed rate-move units.
       rule    a Rule instead of the network: the hardcoded references.
       record  also return the play, per game and round (numpy, detached)."""
    fee = MM_FEE if fee is None else fee
    market = spec.params
    n_games, n_rounds, side = rate_moves_sd.shape[0], spec.rounds, spec.side
    capital = game_capital_pounds(spec)
    # the pound-SELLER is filled when the level is ABOVE the quote, the
    # buyer when below (torch_train's convention, seen from the MM's side)
    fill_direction = 1.0 if side == "A" else -1.0

    #traders state
    held_amount = torch.full((n_games,), float(spec.L))   # start currency
    banked_amount = torch.zeros(n_games)                  # target currency
    #game state
    last_revealed_rate = torch.full((n_games,), float(market.a0))
    recent_moves = torch.zeros(n_games, T.LAGS) if T.LAGS else None
    # the whole hidden rate path for every game, drawn up front. These are
    # the SIMULATOR's variables -- neither network is shown them
    hidden_rates = market.a0 + torch.cumsum(
        market.sd * rate_moves_sd[:, :n_rounds], dim=1)
    settlement_rate = hidden_rates[:, -1] + market.sd * rate_moves_sd[:, n_rounds]

    #MM state
    mm_pounds = torch.zeros(n_games)
    mm_dollars = torch.zeros(n_games)

    log = {key: [] for key in RECORD_KEYS} if record else None

    for round in range(n_rounds):
        rounds_remaining_fraction = (n_rounds - round) / n_rounds
        # this round's true rate: the MM knows it now, the trader after the
        # verdict
        hidden_rate_now = hidden_rates[:, round]

        # ---- offer phase: the FROZEN trader quotes, the MM decides ----------------
        quoted_price, offered_fraction = T.squash(
            trader_net(T.features(rounds_remaining_fraction, 0.0, held_amount,
                                  banked_amount, last_revealed_rate, spec,
                                  recent_moves)),
            last_revealed_rate)
        offered_amount = offered_fraction * held_amount
        if rule is not None:
            fill_weight, break_even = rule_verdict(rule, quoted_price,
                                                   hidden_rate_now, side, fee)
        else:
            # the MM sets its level from what it knows BEFORE reading the
            # offer, then the offer is judged against it
            break_even, _ = mm_squash(
                mm_net(mm_features(rounds_remaining_fraction, 0.0,
                                   hidden_rate_now, mm_pounds, mm_dollars,
                                   offered_amount, spec)),
                hidden_rate_now)
            if tau is None:
                # THE REAL RULE: the market maker either accepts or does not.
                # 1.0 or 0.0, nothing between. Used for all evaluation.
                fill_weight = (fill_direction
                               * (break_even - quoted_price) > 0).double() #.double() converts t/f to 1/0
            else:
                # THE TRAINING SUBSTITUTE: the same rule softened into a ramp,
                # so the payoff has a slope and autodiff has something to
                # follow. 
                move_scale = ((hidden_rate_now - last_revealed_rate) #this not leakage as it is detatched so gradient passes through it 
                              .detach().std().clamp_min(1e-12))
                ramp_width = tau if tau_abs else tau * move_scale
                #the MM's verdict on the offer
                fill_weight = torch.sigmoid(
                    fill_direction * (break_even - quoted_price) / ramp_width)
        if T.LAGS:
            # the rate is revealed to the trader at this point in the round:
            # push the move onto its history window, as torch_train does
            recent_moves = torch.cat(
                [(hidden_rate_now - last_revealed_rate).detach().unsqueeze(1),
                 recent_moves[:, :-1]], dim=1)
        else:
            recent_moves = None

        filled_amount = fill_weight * offered_amount
        held_amount = held_amount - filled_amount
        banked_amount = banked_amount + (filled_amount * quoted_price
                                         if side == "A"
                                         else filled_amount / quoted_price)
        # the same trade from the MM's end: it takes what the trader sold and
        # pays what the trader banked
        if side == "A":
            mm_pounds = mm_pounds + filled_amount
            mm_dollars = mm_dollars - filled_amount * quoted_price
        else:
            mm_dollars = mm_dollars + filled_amount
            mm_pounds = mm_pounds - filled_amount / quoted_price
        if record:
            for key, value in (("quoted_price", quoted_price),
                               ("offered_amount", offered_amount),
                               ("hidden_rate", hidden_rate_now),
                               ("break_even", break_even),
                               ("fill_weight", fill_weight),
                               ("trader_held_after_fill", held_amount),
                               ("trader_banked_after_fill", banked_amount),
                               ("mm_pounds_after_fill", mm_pounds),
                               ("mm_dollars_after_fill", mm_dollars)):
                log[key].append(value.detach().numpy().copy())

        # ---- Bureau phase: rate revealed; the trader first, then the MM ----
        _, bdc_fraction = T.squash(
            trader_net(T.features(rounds_remaining_fraction, 1.0, held_amount,
                                  banked_amount, hidden_rate_now, spec,
                                  recent_moves)),
            hidden_rate_now)
        converted_amount = bdc_fraction * held_amount
        bdc_rate_after_fee = (1.0 - market.bdc_fee) * (
            hidden_rate_now if side == "A" else 1.0 / hidden_rate_now)
        held_amount = held_amount - converted_amount
        banked_amount = banked_amount + converted_amount * bdc_rate_after_fee
        # the MM repays a fraction of its short currancy at the revealed rate
        if rule is not None:
            repay_fraction = torch.full((n_games,), float(rule.repay))
        else:
            _, repay_fraction = mm_squash(
                mm_net(mm_features(rounds_remaining_fraction, 1.0,
                                   hidden_rate_now, mm_pounds, mm_dollars,
                                   torch.zeros(n_games), spec)),
                hidden_rate_now)
        mm_pounds, mm_dollars = mm_bureau(mm_pounds, mm_dollars, repay_fraction,
                                          hidden_rate_now, fee)
        if record:
            for key, value in (("repay_fraction", repay_fraction),
                               ("trader_held_after_bureau", held_amount),
                               ("trader_banked_after_bureau", banked_amount),
                               ("mm_pounds_after_bureau", mm_pounds),
                               ("mm_dollars_after_bureau", mm_dollars)):
                log[key].append(value.detach().numpy().copy())

        # the revealed rate becomes next round's anchor
        last_revealed_rate = hidden_rate_now

    # ---- settlement: the trader's (torch_train's terminal equation) --------
    settlement_factor = (1.0 / settlement_rate if side == "A"
                         else settlement_rate)
    shortfall = torch.clamp(spec.T - banked_amount, min=0.0)
    final_wealth = (held_amount * (1.0 - spec.A)
                    + (banked_amount - spec.B * shortfall) * settlement_factor)
    trader_pl = (final_wealth - spec.L) / spec.L

    # ---- round 5: the MM MUST clear any short at the round-5 rate, same
    # fee; what is left is scored in pounds at that rate with no fee. 
    mm_pounds, mm_dollars = mm_bureau(mm_pounds, mm_dollars, torch.ones(n_games),
                                      settlement_rate, fee)
    mm_pl = (mm_pounds + mm_dollars / settlement_rate) / capital

    if record:
        log = {key: np.stack(value, axis=1) for key, value in log.items()}
        log["settlement_rate"] = settlement_rate.detach().numpy().copy()
        log["mm_pounds_end"] = mm_pounds.detach().numpy().copy()
        log["mm_dollars_end"] = mm_dollars.detach().numpy().copy()
    return mm_pl, trader_pl, log


def pl_loss(mm_net, trader_net, spec, rate_moves_sd, tau):
    #negation is because optimisers MINIMISE and we want to MAXIMISE the payoff
    mm_pl, _, _ = rollout(mm_net, trader_net, spec, rate_moves_sd, tau=tau)
    return -mm_pl.mean()


def operating_threshold(price_over_rate, volume, fill_weight, side):
    """The single P/X cut that best reproduces the recorded VERDICTS, by
    offered volume, with the volume it misclassifies. This, not the emitted
    level averaged over offers, is the policy's threshold: away from the
    margin the verdict does not constrain the level, so its value there is
    unlearned drift (measured: side A agreeing with the naive rule on 100%
    of volume while the emitted mean sat 0.4% below the true rate)."""
    order = np.argsort(price_over_rate)
    u = price_over_rate[order]
    accepted = (volume * fill_weight)[order]
    rejected = (volume * (1.0 - fill_weight))[order]
    # cumulative volume strictly below each cut point (cut k sits before u[k])
    accepted_below = np.concatenate([[0.0], np.cumsum(accepted)])
    rejected_below = np.concatenate([[0.0], np.cumsum(rejected)])
    if side == "A":     # accept iff u < t: misses = rejects below + accepts above
        missed = rejected_below + (accepted_below[-1] - accepted_below)
    else:               # accept iff u > t: misses = accepts below + rejects above
        missed = accepted_below + (rejected_below[-1] - rejected_below)
    k = int(np.argmin(missed))
    edges = np.concatenate([[u[0]], (u[1:] + u[:-1]) / 2.0, [u[-1]]])
    return float(edges[k]), float(missed[k] / (volume.sum() + 1e-300))


def repay_on_short_books(log):
    """The Bureau head as OPERATED: the repay fraction averaged over the
    (game, round) cells that actually hold a short, weighted by the short's
    pound value. The plain mean over all games says little -- most books
    have nothing short and the head's output there does nothing."""
    short_value = (np.clip(-log["mm_pounds_after_fill"], 0.0, None)
                   + np.clip(-log["mm_dollars_after_fill"], 0.0, None)
                   / log["hidden_rate"])
    weighted = (short_value * log["repay_fraction"]).sum(axis=0)
    total = short_value.sum(axis=0)
    return np.where(total > 0, weighted / np.maximum(total, 1e-300), np.nan)


def volume_agreement(log, side, benchmark_rule, fee=None):
    """How much of the offered VOLUME the recorded verdicts agree with the
    BENCHMARK rule on, and the filled share of that volume for both. Volume-
    weighted so the trader's zero-size leftovers (sigmoid never reaches 1)
    cannot count as decisions.
      The benchmark is the BEST-SCORING reference rule, passed in, not a
    guess from the fee. Deriving it from the fee was wrong on side B at 1%,
    where the best rule is naive+CARRY: a carrier only meets the fee at the
    round-5 clear, so its threshold stays at the true rate and scoring its
    verdicts against the fee-aware cut read as a 1.4% error rate when the
    MM was right. Returns the name so the print cannot mislabel it."""
    fee = MM_FEE if fee is None else fee
    break_even = log["hidden_rate"]
    name = "naive"
    if benchmark_rule.fee_aware:
        break_even = ((1.0 - fee) * break_even if side == "A"
                      else break_even / (1.0 - fee))
        name = "fee-aware"
    benchmark = mm_accepts(log["quoted_price"], break_even, side).astype(float)
    volume = log["offered_amount"]
    total = volume.sum()
    agree = (volume * (log["fill_weight"] == benchmark)).sum() / total
    filled = (volume * log["fill_weight"]).sum() / total
    filled_benchmark = (volume * benchmark).sum() / total
    return agree, filled, filled_benchmark, name


# ---------------------------------------------------------------------------
# 3. checks -- the risk a reimplementation creates, discharged before training
# ---------------------------------------------------------------------------
CHECK_FEE = 0.01     # the checks exercise the fee arithmetic whatever MM_FEE is


def check_trader_game_matches_torch_train(spec, trader_net, n_games=2000, seed=3):
    """check 1 -- the trader's half of this file's game IS torch_train's.
    Under the naive rule the MM here gives exactly the verdicts of
    torch_train's hard gate, so the trader's per-game P/L must agree to
    floating point, not to a tolerance. Runs BEFORE training and raises on
    failure: a trainer built on a subtly wrong copy of the trader's game
    would optimise beautifully and mean nothing."""
    generator = torch.Generator().manual_seed(seed)
    rate_moves_sd = torch.randn(n_games, spec.rounds + 1, generator=generator)
    with torch.no_grad():
        reference_pl = T.rollout(trader_net, spec, rate_moves_sd, tau=None)
        _, trader_pl, _ = rollout(None, trader_net, spec, rate_moves_sd,
                                  rule=Rule("naive+carry", False, 0.0))
    max_gap = float((trader_pl - reference_pl).abs().max())
    passed = max_gap < 1e-9
    print(f"  check 1  trader P/L here vs torch_train.rollout, {n_games} games "
          f"(side {spec.side}): max |gap| {max_gap:.2e}  "
          f"[{'ok' if passed else 'FAIL'}]", flush=True)
    assert passed, "the trader's game here does not match torch_train"


def check_books_conserve(spec, trader_net, n_games=2000, seed=5):
    """check 2 -- the seam this file adds. Two players are audited at
    CHECK_FEE, chosen because between them they reach every Bureau branch:
    the naive rule flattening every round (it fills, and at a fee some of
    those fills flatten into a short of the OTHER leg), and an untrained net
    whose level is set generous here so that it fills and its repay head is
    live -- the training init rejects everything, which would leave the
    Bureau arithmetic untested. In both, the recorded play must show: fills
    are 0 or 1; every fill moves the trader's and the MM's pounds and
    dollars by equal and opposite amounts; every Bureau trade moves value at
    exactly the fee-adjusted rate; no repay deepens a short or turns a long
    leg short; and round 5 leaves no short the book could have covered."""
    side, f = spec.side, CHECK_FEE
    fill_direction = 1.0 if side == "A" else -1.0
    generator = torch.Generator().manual_seed(seed)
    rate_moves_sd = torch.randn(n_games, spec.rounds + 1, generator=generator)

    # the check's own net: same architecture, level moved to the generous
    # side so it accepts nearly everything. A check net, not a training one
    generous_net = build_mm_net(SEED_INIT, side)
    with torch.no_grad():
        generous_net.head.bias[0] = torch.logit(
            torch.tensor((1.0 + fill_direction * LEVEL_INIT_OFFSET) / 2.0))

    for label, kwargs in (("naive+flatten", {"rule": Rule("naive+flatten", False, 1.0)}),
                          ("untrained net", {"mm_net_override": generous_net})):
        mm_net = kwargs.pop("mm_net_override", None)
        with torch.no_grad():
            _, _, log = rollout(mm_net, trader_net, spec, rate_moves_sd,
                                tau=None, fee=f, record=True, **kwargs)
        fill = log["fill_weight"]
        assert np.all((fill == 0.0) | (fill == 1.0)), "hard verdict gave a fraction"
        filled_share = ((log["offered_amount"] * fill).sum()
                        / log["offered_amount"].sum())

        # the trader's PHYSICAL pounds and dollars, from its (held, banked) roles
        def physical(held, banked):
            return (held, banked) if side == "A" else (banked, held)
        # start-of-round books: the previous round's post-Bureau state
        start_held = np.concatenate([np.full((n_games, 1), spec.L),
                                     log["trader_held_after_bureau"][:, :-1]], axis=1)
        start_banked = np.concatenate([np.zeros((n_games, 1)),
                                       log["trader_banked_after_bureau"][:, :-1]], axis=1)
        start_pounds = np.concatenate([np.zeros((n_games, 1)),
                                       log["mm_pounds_after_bureau"][:, :-1]], axis=1)
        start_dollars = np.concatenate([np.zeros((n_games, 1)),
                                        log["mm_dollars_after_bureau"][:, :-1]], axis=1)
        trader_pounds_0, trader_dollars_0 = physical(start_held, start_banked)
        trader_pounds_1, trader_dollars_1 = physical(log["trader_held_after_fill"],
                                                     log["trader_banked_after_fill"])
        gap_fill = max(
            np.abs((trader_pounds_1 - trader_pounds_0)
                   + (log["mm_pounds_after_fill"] - start_pounds)).max(),
            np.abs((trader_dollars_1 - trader_dollars_0)
                   + (log["mm_dollars_after_fill"] - start_dollars)).max())

        # the MM's Bureau trades: sold currency x (1-f) x rate == bought currency
        d_pounds = log["mm_pounds_after_bureau"] - log["mm_pounds_after_fill"]
        d_dollars = log["mm_dollars_after_bureau"] - log["mm_dollars_after_fill"]
        rate = log["hidden_rate"]
        sold_pounds = d_pounds < 0
        gap_bureau = max(
            np.abs(np.where(sold_pounds, d_pounds * (1 - f) * rate + d_dollars, 0.0)).max(),
            np.abs(np.where(~sold_pounds, d_dollars * (1 - f) / rate + d_pounds, 0.0)).max())

        # a repay may only shrink a short, and may never turn a long short
        before = (log["mm_pounds_after_fill"], log["mm_dollars_after_fill"])
        after = (log["mm_pounds_after_bureau"], log["mm_dollars_after_bureau"])
        deepest_before = np.minimum(np.minimum(before[0], 0.0), np.minimum(before[1], 0.0))
        deepest_after = np.minimum(np.minimum(after[0], 0.0), np.minimum(after[1], 0.0))
        assert np.all(deepest_after >= deepest_before - 1e-6), "a repay deepened a short"
        for was, now in zip(before, after):
            assert not np.any((was >= -1e-6) & (now < -1e-6)), \
                "a repay turned a long leg short"

        # round 5: no short is left that the other leg could have covered.
        # Measured in POUNDS -- the coverable part of whichever leg is short
        # -- so the tolerance means the same thing as everywhere else
        end_pounds, end_dollars = log["mm_pounds_end"], log["mm_dollars_end"]
        a5 = log["settlement_rate"]
        gap_clear = max(
            np.minimum(np.clip(-end_pounds, 0, None),
                       np.clip(end_dollars, 0, None) / a5).max(),
            np.minimum(np.clip(-end_dollars, 0, None) / a5,
                       np.clip(end_pounds, 0, None)).max())

        passed = gap_fill < 1e-6 and gap_bureau < 1e-6 and gap_clear < 1e-6
        print(f"  check 2  {label:<14} books conserve, {n_games} games "
              f"(side {side}, fee {f}, filled {filled_share:.1%} of volume): "
              f"fill gap {gap_fill:.2e}  Bureau gap {gap_bureau:.2e}  round-5 "
              f"gap {gap_clear:.2e} GBP  [{'ok' if passed else 'FAIL'}]", flush=True)
        assert passed, "the MM's book does not conserve value"
        assert filled_share > 0.01, (
            f"check 2's {label} player filled almost nothing -- the Bureau "
            f"arithmetic would go untested")


def check_rule_pl_closed_form(spec, trader_net, n_games=2000, seed=7):
    """check 3 -- the MM's P/L under a hardcoded rule, replayed, must equal
    the closed form summed from the recorded fills. Two rules at CHECK_FEE:
    fee-aware+flatten (every fill is repaid the same round and leaves a
    non-negative surplus, so no cap can bind) and naive+carry (nothing is
    repaid until round 5, where the settle CAN be capped). Between them the
    solvent and the insolvent settle are both covered; check 2 covers the
    per-round Bureau arithmetic elementwise. The recorded verdicts are also
    recomputed from (P, X)."""
    generator = torch.Generator().manual_seed(seed)
    rate_moves_sd = torch.randn(n_games, spec.rounds + 1, generator=generator)
    side, g = spec.side, CHECK_FEE
    capital = game_capital_pounds(spec)
    worst = 0.0
    for rule in (Rule("fee-aware+flatten", True, 1.0),
                 Rule("naive+carry", False, 0.0)):
        with torch.no_grad():
            mm_pl, _, log = rollout(None, trader_net, spec, rate_moves_sd,
                                    rule=rule, fee=g, record=True)
        P, X, q = log["quoted_price"], log["hidden_rate"], log["offered_amount"]
        a5 = log["settlement_rate"]
        break_even = X if not rule.fee_aware else (
            (1 - g) * X if side == "A" else X / (1 - g))
        verdict = mm_accepts(P, break_even, side).astype(float)
        assert np.array_equal(verdict, log["fill_weight"]), "verdict mismatch"
        filled = verdict * q
        if rule.repay == 1.0:                   # repaid the same round
            if side == "A":     # took pounds, bought the dollars back at X
                closed = (filled * (1.0 - P / ((1 - g) * X))).sum(axis=1) / capital
            else:               # took dollars, bought the pounds back at X
                closed = ((filled * (1.0 - X / ((1 - g) * P))).sum(axis=1)
                          / a5 / capital)
        else:                                   # carried, cleared at round 5
            # the whole book arrives at round 5 and settles once, and the
            # settle is CAPPED by the long leg: a carry book can arrive
            # insolvent (its fills were priced against X_n, and a5 may have
            # moved far from those), in which case it clears what it can
            # afford and the residue is scored directly
            if side == "A":
                long_leg, short_leg = filled.sum(axis=1), (filled * P).sum(axis=1)
                bought = np.minimum(short_leg, long_leg * (1 - g) * a5)
                pounds = long_leg - bought / ((1 - g) * a5)
                dollars = bought - short_leg
            else:
                long_leg, short_leg = filled.sum(axis=1), (filled / P).sum(axis=1)
                bought = np.minimum(short_leg, long_leg * (1 - g) / a5)
                pounds = bought - short_leg
                dollars = long_leg - bought * a5 / (1 - g)
            closed = (pounds + dollars / a5) / capital
        gap = float(np.abs(mm_pl.numpy() - closed).max())
        worst = max(worst, gap)
        print(f"  check 3  {rule.name:<18} replayed vs closed form, {n_games} "
              f"games (side {side}, fee {g}): max |gap| {gap:.2e}  "
              f"[{'ok' if gap < 1e-9 else 'FAIL'}]", flush=True)
    assert worst < 1e-9, "the MM's replayed P/L does not match its closed form"


# ---------------------------------------------------------------------------
# 4. training -- initialise, sample, average, backward, step
# ---------------------------------------------------------------------------
def train(spec, trader_net, run):
    mm_net = build_mm_net(SEED_INIT, spec.side)
    optimiser = torch.optim.AdamW(mm_net.parameters(), lr=LR)

    n_anneal = max(int(TAU_ANNEAL_FRAC * ITERS), 1)
    decay = (TAU_END / TAU_START) ** (1.0 / max(n_anneal - 1, 1))

    # the VALIDATION set: one fixed batch of paths, reused at every check so
    # checkpoints are compared on identical games. Its own seed, so it never
    # overlaps the training draws below.
    validation_generator = torch.Generator().manual_seed(SEED_PATH + 1)
    val_rate_moves_sd = torch.randn(VAL_PATHS, spec.rounds + 1,
                                    generator=validation_generator)
    # the hardcoded rules on those same paths: curves reference lines
    val_references, val_reference_se = {}, {}
    with torch.no_grad():
        for rule in reference_rules(MM_FEE):
            rule_pl = rollout(None, trader_net, spec, val_rate_moves_sd,
                              rule=rule)[0]
            val_references[rule.name] = float(rule_pl.mean())
            # how firm the line is on THESE paths. The carry rules ride the
            # whole book to round 5 and inherit the settlement rate's
            # variance, so their lines are ~5x softer than the flatten ones
            # (+/- 2-3 bp against +/- 0.6). mm_diagnostics bands them, and
            # convergence is judged on the PAIRED eval, never off the figure.
            val_reference_se[rule.name] = float(
                2 * rule_pl.std() / np.sqrt(VAL_PATHS))
    print("  hardcoded rules on the validation paths:  " + "   ".join(
        f"{name} {value * 1e4:+.2f} +/- {val_reference_se[name] * 1e4:.2f} bp"
        for name, value in val_references.items()), flush=True)
    # the verdicts are scored against the best of them, whichever that is
    benchmark_rule = max(reference_rules(MM_FEE),
                         key=lambda rule: val_references[rule.name])

    best_checkpoint = {"hard_pl": -1e9, "iteration": -1}
    curve_iterations, curve_hard_pl, curve_agree = [], [], []
    best_reference = max(val_references.values())
    for iteration in range(ITERS):
        # ramp width for this step: anneal, then hold (see TAU_ANNEAL_FRAC)
        if GATE == "hard":
            tau = None
        elif iteration < n_anneal:
            tau = TAU_START * decay ** iteration      # shrinking
        else:
            tau = TAU_END                             # held

        # STEP 1: draw a FRESH batch of games. A new seed every iteration, so
        # no game is ever seen twice and there is nothing to memorise.
        batch_generator = torch.Generator().manual_seed(
            SEED_PATH + 10_000 + iteration)
        batch_rate_moves_sd = torch.randn(BATCH, spec.rounds + 1,
                                          generator=batch_generator)

        # STEP 2: play them all with the current weights, average the MM P/L
        loss = pl_loss(mm_net, trader_net, spec, batch_rate_moves_sd, tau)
        if not torch.isfinite(loss):
            print(f"  [guard] loss non-finite at iter {iteration}. "
                  f"Skipping step.", flush=True)
            optimiser.zero_grad()
            continue

        # STEP 3: backprop 
        optimiser.zero_grad()
        loss.backward()

        # STEP 4: cap the step's size, then take it
        torch.nn.utils.clip_grad_norm_(mm_net.parameters(), CLIP_NORM)
        optimiser.step()

        # STEP 5: periodically score the MM on the TRUE game (hard verdicts)
        # on the fixed validation paths, and keep the best version so far
        if iteration % VAL_EVERY == 0 or iteration == ITERS - 1:
            with torch.no_grad():
                hard_pl, _, log = rollout(mm_net, trader_net, spec,
                                          val_rate_moves_sd, tau=None,
                                          record=True)
            hard_game_pl = float(hard_pl.mean())
            agree, filled, filled_benchmark, _ = volume_agreement(
                log, spec.side, benchmark_rule)
            curve_iterations.append(iteration)
            curve_hard_pl.append(hard_game_pl)
            curve_agree.append(agree)
            tau_label = "  hard " if tau is None else f"tau {tau:8.5f}"
            # 'batch P/L' is a 1,000-path estimate of the SMOOTHED game and
            # is noisy by design; 'hard P/L' is 20,000 paths of the real one.
            # 'agree' is the offered volume on which the verdicts match the
            # BENCHMARK rule (naive at fee 0, fee-aware above it); 'filled'
            # the volume taken, learner / benchmark
            print(f"  iter {iteration:>5}  {tau_label}  "
                  f"batch P/L {-loss.item() * 1e4:+9.2f} bp  "
                  f"hard P/L {hard_game_pl * 1e4:+9.2f} bp  "
                  f"(best rule {best_reference * 1e4:+8.2f})  "
                  f"agree {agree * 100:5.1f}%  filled {filled * 100:5.1f}% / "
                  f"{filled_benchmark * 100:5.1f}%", flush=True)
            if hard_game_pl > best_checkpoint["hard_pl"]:
                best_checkpoint.update(hard_pl=hard_game_pl,
                                       iteration=iteration)
                torch.save(mm_net.state_dict(), run / "mm_best.pth")
    torch.save(mm_net.state_dict(), run / "mm_final.pth")

    # early stopping: evaluate the checkpoint that validated best on the
    # HARD game (final kept on disk)
    print(f"  best hard-game validation {best_checkpoint['hard_pl'] * 1e4:+.2f} bp "
          f"at iter {best_checkpoint['iteration']}; evaluating mm_best.pth "
          f"(final kept on disk)", flush=True)
    mm_net.load_state_dict(torch.load(run / "mm_best.pth"))
    return (mm_net, best_checkpoint, (val_references, val_reference_se,
            benchmark_rule), (curve_iterations, curve_hard_pl, curve_agree))


# ---------------------------------------------------------------------------
# 5. main -- side A, then side B; every number a paired hard-game replay
# ---------------------------------------------------------------------------
def paired(learner_pl, rule_pl):
    #learner minus rule on the SAME paths: mean and 2 standard errors of the
    #paired difference (unpaired errors make every comparison a tie)
    difference = (learner_pl - rule_pl).numpy()
    return difference.mean(), 2.0 * difference.std(ddof=1) / np.sqrt(len(difference))


def run_side(side):
    # ---- 1. WHICH GAME ----------------------------------------------------
    #Runs one side end to end — checks, training, evaluation, artefacts — and
    #main() does side A then side B.
    spec, trader_net, fingerprint = load_trader(side)
    rules = reference_rules(MM_FEE)
    run = results_path(f"mm_phase1/mm_{side}_fee{MM_FEE:g}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "trader_fingerprint").write_text(fingerprint)

    print(f"\n################ side {side}: card L={spec.L:,.0f} T={spec.T:,.0f} "
          f"A={spec.A} B={spec.B} rounds={spec.rounds} K={spec.K}  MM fee "
          f"{MM_FEE} ################", flush=True)
    print(f"  frozen trader {fingerprint} from mm_phase1/trader_{side}")
    print(f"  sample budget: {ITERS:,} iters x {BATCH:,} games = "
          f"{ITERS * BATCH:,} training games")
    print("  method: direct policy optimisation through a differentiable "
          "simulator with the trader frozen in the loop -- NOT model-free RL")
    print(f"  verdict: private level anchored on the true rate, started "
          f"{LEVEL_INIT_OFFSET:.0%} off on the conservative side")
    if GATE == "hard":
        print("  GATE=hard: the level enters only through an indicator, so "
              "its gradient is zero -- kept to show it")
    else:
        print(f"  GATE=smoothed: tau anneal {TAU_START} -> {TAU_END} x the "
              f"observed innovation scale, measured per batch (no "
              f"distribution constant anywhere; adapts to any sd)")

    # ---- 2. THE CHECKS ----------------------------------------------------
    if CHECKS:
        check_trader_game_matches_torch_train(spec, trader_net)
        check_books_conserve(spec, trader_net)
        check_rule_pl_closed_form(spec, trader_net)
    start_time = time.time()

    # ---- 3. TRAINING ------------------------------------------------------
    mm_net, best_checkpoint, references, curve = train(spec, trader_net, run)
    val_references, val_reference_se, benchmark_rule = references

    # ---- 4. EVALUATION ----------------------------------------------------
    # A large fresh sample (never seen in training or validation), played on
    # the TRUE game: learner and every rule on the SAME paths
    eval_generator = torch.Generator().manual_seed(SEED_PATH + 2)
    eval_rate_moves_sd = torch.randn(EVAL_PATHS, spec.rounds + 1,
                                     generator=eval_generator)
    with torch.no_grad():
        eval_mm_pl, eval_trader_pl, _ = rollout(mm_net, trader_net, spec,
                                                eval_rate_moves_sd, tau=None)
        eval_rule_mm_pl, eval_rule_trader_pl = [], []
        for rule in rules:
            rule_mm_pl, rule_trader_pl, _ = rollout(None, trader_net, spec,
                                                    eval_rate_moves_sd, rule=rule)
            eval_rule_mm_pl.append(rule_mm_pl)
            eval_rule_trader_pl.append(rule_trader_pl)
        # a smaller fresh replay WITH the play recorded, for mm_diagnostics
        record_generator = torch.Generator().manual_seed(SEED_PATH + 3)
        _, _, log = rollout(mm_net, trader_net, spec,
                            torch.randn(RECORD_PATHS, spec.rounds + 1,
                                        generator=record_generator),
                            tau=None, record=True)
    agree, filled, filled_benchmark, benchmark_name = volume_agreement(
        log, side, benchmark_rule)
    # the policy as OPERATED, per round: the threshold fitted to the verdicts
    # (identified only where the trader offers real volume) and the repay on
    # books that actually hold a short
    volume = log["offered_amount"]
    price_over_rate = log["quoted_price"] / log["hidden_rate"]
    threshold_by_round, stray_by_round = zip(*[
        operating_threshold(price_over_rate[:, r], volume[:, r],
                            log["fill_weight"][:, r], side)
        if volume[:, r].sum() > 0 else (np.nan, np.nan)
        for r in range(spec.rounds)])
    repay_by_round = repay_on_short_books(log)
    volume_share = volume.sum(axis=0) / volume.sum()

    two_se = 2 * float(eval_mm_pl.std()) / np.sqrt(EVAL_PATHS)
    print(f"\nside {side} final (hard game, {EVAL_PATHS:,} paths, torch engine):")
    print(f"  learned MM P/L {float(eval_mm_pl.mean()) * 1e4:+.2f} bp "
          f"+/- {two_se * 1e4:.2f} of the game's capital "
          f"(GBP {game_capital_pounds(spec):,.0f})")
    for rule, rule_mm_pl in zip(rules, eval_rule_mm_pl):
        gap, gap_2se = paired(eval_mm_pl, rule_mm_pl)
        print(f"  vs {rule.name:<18} {float(rule_mm_pl.mean()) * 1e4:+.2f} bp   "
              f"learner - rule {gap * 1e4:+.2f} +/- {gap_2se * 1e4:.2f} bp (paired)")
    print(f"  verdicts agree with the {benchmark_name} rule on {agree * 100:.1f}% "
          f"of offered volume; filled {filled * 100:.1f}% "
          f"({benchmark_name} {filled_benchmark * 100:.1f}%)")
    print("  operating threshold P/X by round, fitted to the verdicts (share "
          "of offered volume in brackets): " + "  ".join(
              f"r{r + 1} {threshold_by_round[r]:.4f} "
              f"({volume_share[r] * 100:.0f}%)"
              for r in range(spec.rounds))
          + f"   [a single cut strays from the verdicts on "
          f"{np.nansum(np.array(stray_by_round) * volume_share) * 100:.2f}% "
          f"of volume]")
    print("  Bureau repay fraction on SHORT books, short-weighted, by round: "
          + "  ".join(f"r{r + 1} {repay_by_round[r]:.3f}"
                      for r in range(spec.rounds)))
    print(f"  trader P/L: vs learned MM {float(eval_trader_pl.mean()) * 100:+.4f}%   "
          + "   ".join(f"vs {rule.name} {float(pl.mean()) * 100:+.4f}%"
                       for rule, pl in zip(rules, eval_rule_trader_pl)))

    # ---- 5. ARTEFACTS -----------------------------------------------------
    # everything mm_diagnostics needs, and nothing it must recompute
    np.savez(run / "mm_eval.npz",
             side=side, fee=MM_FEE, see_offer_size=SEE_OFFER_SIZE,
             tau=np.array([TAU_START, TAU_END, TAU_ANNEAL_FRAC]),
             level_init_offset=LEVEL_INIT_OFFSET,
             card=np.array([spec.L, spec.T, spec.A, spec.B]),
             a0=spec.params.a0, sd=spec.params.sd,
             trader_bdc_fee=spec.params.bdc_fee,
             capital_pounds=game_capital_pounds(spec),
             trader_fingerprint=fingerprint,
             rule_names=np.array([rule.name for rule in rules]),
             rule_repay=np.array([rule.repay for rule in rules]),
             val_rule_pl=np.array([val_references[rule.name] for rule in rules]),
             val_rule_se=np.array([val_reference_se[rule.name] for rule in rules]),
             benchmark_name=benchmark_name, benchmark_rule=benchmark_rule.name,
             curve_iterations=np.array(curve[0]),
             curve_hard_pl=np.array(curve[1]),
             curve_agree=np.array(curve[2]),
             best_iteration=best_checkpoint["iteration"],
             best_val_pl=best_checkpoint["hard_pl"],
             eval_mm_pl=eval_mm_pl.numpy(),
             eval_trader_pl=eval_trader_pl.numpy(),
             eval_rule_mm_pl=np.stack([pl.numpy() for pl in eval_rule_mm_pl]),
             eval_rule_trader_pl=np.stack([pl.numpy() for pl in eval_rule_trader_pl]),
             **{f"record_{key}": value for key, value in log.items()})
    print(f"artefacts in {run}/  ({(time.time() - start_time) / 60:.2f} min)",
          flush=True)
    return {"side": side, "learner": float(eval_mm_pl.mean()),
            "rules": {rule.name: float(pl.mean())
                      for rule, pl in zip(rules, eval_rule_mm_pl)},
            "agree": agree, "benchmark": benchmark_name,
            "threshold": threshold_by_round}


def main():
    summaries = [run_side(side) for side in SIDES]
    print("\n" + "=" * 72)
    print(f"LEARNED MM, fee {MM_FEE}, hard game, {EVAL_PATHS:,} paths per side "
          f"(bp of the game's capital)")
    for s in summaries:
        print(f"  side {s['side']}: learner {s['learner'] * 1e4:+8.2f}   "
              + "   ".join(f"{name} {value * 1e4:+8.2f}"
                           for name, value in s["rules"].items())
              + f"   agree({s['benchmark']}) {s['agree'] * 100:.1f}%   round-1 threshold "
              f"{s['threshold'][0]:.4f} x true rate")
    print("  validation read: side A should flatten on naive+flatten (its "
          "exact optimum at fee 0); side B should reach naive+carry -- "
          "holding dollars to round 5 gains from E[1/a5] > 1/X (Siegel)")


if __name__ == "__main__":
    main()