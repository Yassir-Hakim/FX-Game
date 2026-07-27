"""
================================================================================
 VERSION 2
 One trader, the whole game: four trading rounds, K sequential (P, q) offers
 per round, carry-forward between rounds, and the official P/L settled at the
 round-5 rate with residue and deficit penalties.
================================================================================

 Structure:
   1. THE TRADER AND THE CLOCK  - parameters; what one game of v2 is
   2. GRIDS                     - offer grid, quadrature, (c, d, a) grids
   3. THE TERMINAL LAYER        - the official P/L and the kink at the target
   4. THE ROUND SOLVER          - interval beliefs, the (P, q) ladder, the BdC
   5. THE FULL SOLVER           - stitch rounds R -> 1 by backward induction
   6. ACTING FROM THE TABLES    - greedy actions, used by replay and plots
   7. POLICY REPLAY             - Monte Carlo through the shared rules
   8. THE CLAIRVOYANT BENCHMARK - per-path hindsight optimum; regret
   9. VALIDATION                - gate 1 (v1 anchor), gate 2 (B=0 linearity),
                                  gate 3 (MC vs DP), gate 4 (money conservation)
  10. PLOTS
  11. RUN

 State and beliefs, in one paragraph. Across rounds the trader's state is
 (round n, pounds c, dollars d, last revealed rate a): the deficit penalty puts
 a concave kink in the terminal reward at the dollar target T, so dollars
 banked stop being a passive tally and join the state. Within a round the
 hidden rate X ~ N(a, sd^2) and every offer updates an INTERVAL belief: a
 rejection at price P caps X (ceiling, as in v1); an acceptance with q < c
 floors X and the trader keeps selling -- floors are new in v2 because v1's
 all-or-nothing accept ended the round. Brackets die at the round boundary
 when the BdC reveals X, so rounds are linked only through (c, d, a).
================================================================================
"""
import math
import time
import numpy as np
from dataclasses import dataclass, field
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from fx_mechanics import (
    GameParams,
    mm_accepts,
    trade_rate,
    rate_units,
    bdc_payoff_per_unit,
    A0_DEFAULT,
    sd_DEFAULT,
    BDC_FEE_DEFAULT,
)

NO_FLOOR = -1          # sentinel floor index: "no floor learned yet"  (z = -inf)
                     # the ceiling sentinel is n_offer: "no ceiling yet" (+inf)

# ==============================================================================
# 1. THE TRADER AND THE CLOCK
# ==============================================================================

@dataclass
class TraderSpec:
    """

    L        starting capital, in the INITIAL currency (pounds on side "A",
             dollars on side "B").
    T        the target, in the TARGET currency (dollars on "A", pounds on "B").
    A        residue penalty rate on initial currency still held at the end.
    B        deficit penalty rate on the target-currency shortfall below T.
    rounds   trading rounds (the game's round 5 is MM book-clearing only; for
             the trader it collapses to drawing the settlement rate a5).
    K        sequential offers per round.
    all_or_nothing   gate-1 switch: force q = everything, as in v1.
    terminal_mode    "pounds" is the official P/L; "dollars" is gate 1's
                     objective (maximise expected dollars, exactly v1's).    
    """
    L: float = 100_000.0
    T: float = 125_000.0
    A: float = 0.02
    B: float = 0.03
    rounds: int = 4
    K: int = 3
    params: GameParams = field(default_factory=GameParams)
    all_or_nothing: bool = False
    terminal_mode: str = "pounds"
    q_floor: float = 0.0          
    side: str = "A"               # "A": start pounds, target dollars (T1 family)
                                  # "B": start dollars, target pounds (T4 family)


# ------------------------------------------------------------------------------
#
#   the REFLECTION. Prices live on a z-grid, P = a + sd z. The MM takes a
#   seller's offer when it is BELOW the hidden rate and a buyer's when it is
#   ABOVE, so acceptance floors the rate for one and caps it for the other --
#   and the bracket machinery below hardcodes accept -> floor. Reading the
#   buyer's grid DOWNWARDS, z |-> a - sd z, is the reflection Y = 2a - X: it
#   flips the inequality while leaving the law exactly N(a, sd^2), so Phi, the
#   quadrature and the prefix sums are inherited untouched. Working in the
#   reciprocal rate 1/X instead would NOT do this -- 1/X is skewed, and the
#   normal machinery would be wrong in the tails.
#
# Settlement is the one asymmetry that is not a mirror: the seller converts
# dollars at 1/a5, which is convex and needs kappa = E[1/a5] (Jensen), while the
# buyer converts pounds at a5, which is LINEAR, so the martingale gives
# E[a5 | a4] = a4 exactly and no quadrature is needed. 
# ------------------------------------------------------------------------------

def _rate_at(spec, a, z):
    """The real rate ($/GBP) at grid coordinate z -- an offer price, or a
    quadrature node. The buyer reads the grid downwards (the reflection)."""
    if spec.side == "B":
        return a - spec.params.sd * z
    return a + spec.params.sd * z


# ==============================================================================
# 2. GRIDS
# ==============================================================================
#
# Four grids, one honesty note.
#
#   offers   prices live on a z-grid, P = a + sd * z, z in [-zmax, zmax]:
#            the same convention as v1. Acceptance probabilities depend only
#            on z, so the grid is reused at every anchor.

#   quad     expectations over the hidden X use cells BETWEEN offer points
#            (quad_per_cell per interval) plus two tail cells beyond +-zmax,
#            weighted by exact normal mass. Cumulative sums make the
#            conditional expectation over ANY bracket an O(1) lookup, and the
#            cumulative weights reproduce Phi at the offer points exactly.

#   c, d     pounds on a uniform grid 0..L; dollars on a uniform grid with the
#            target T EXACTLY on a node, so the kink is never smeared by
#            interpolation. Offer sizes q and BdC dumps m move along c-grid
#            steps; dollar arrivals are interpolated linearly in d (with
#            linear extrapolation above the last node).

#   a        the anchor (last revealed rate) on a grid a0 +- a_halfwidth sds;
#            continuation values are interpolated linearly in a.
#
# The honesty note: q, m, P and the belief brackets are all quantised. That is
# a declared approximation, controlled by the validation gates and by rerunning
# at finer grids (see the write-up's convergence table).

class Grids:
    def __init__(self, spec, zmax=4.0, n_offer=49, quad_per_cell=5,
                 n_c=17, n_d=41, n_a=13, a_halfwidth_sds=8.0, n_qfrac=24):
        params = spec.params
        self.zmax = zmax
        self.n_offer = n_offer
        self.quad_per_cell = quad_per_cell #how many integration sample points to place between each pair of offer-price grid points

        # ---- offer grid ------------------------------------------------------
        self.z = np.linspace(-zmax, zmax, n_offer)
        self.Phi = norm.cdf(self.z)

        # ---- quadrature cells, aligned to the offer grid ---------------------
       
        #The count is chosen so the boundaries line up exactly with the offer-price grid 
        #(quad_per_cell finer) -- that alignment is what makes the interval lookup below work.
        n_boundaries = (n_offer - 1) * quad_per_cell + 1
        b = np.linspace(-zmax, zmax, n_boundaries)
        
        w_int = np.diff(norm.cdf(b)) #Difference of the normal cdf at rectangle edges
        z_int = 0.5 * (b[:-1] + b[1:]) #Uses midpoint for accuracy

        
        w_tail = norm.cdf(-zmax) #Tail probabilities beyond ±zmax
        z_tail = norm.pdf(zmax)/w_tail #Avg tail price

        # Full sample set: left tail, all interior cells, right tail.
        # rate_samples = the rates; rate_weights = their probabilities.
        # An expectation over the whole rate is now sum(rate_weights * f(...)).
        self.rate_samples = np.concatenate([[-z_tail], z_int, [z_tail]])
        self.rate_weights = np.concatenate([[w_tail], w_int, [w_tail]])
        self.nq = self.rate_samples.size

        # Running total of probability. 
        #Speed trick - expectation over an interval is an instant lookup instead of a dresh sum
        self.cumulative_mass = np.concatenate(
            [[0.0], np.cumsum(self.rate_weights)])

        #Lookup table: for each offer price, which quadrature cell does it sit at
        self.offer_boundary = 1 + quad_per_cell * np.arange(n_offer) 

        # ---- pounds ----------------------------------------------------------
        self.n_c = n_c
        self.c = np.linspace(0.0, spec.L, n_c)
        self.dc = self.c[1] - self.c[0]

        # ---- the offer SIZE as a FRACTION of current holdings.----------------
        base = np.linspace(0.0, 1.0, n_qfrac)
        self.f_all = np.unique(np.concatenate([[0.002, 0.01], base])) #q the solver tries, manually added smaller for probes
        self.f_pos = self.f_all[self.f_all > 1e-12] #>0 only

        # ---- dollars, with the target on a node ------------------------------
        a_top = params.a0 + (a_halfwidth_sds + zmax) * params.sd #highest possible rate
        if spec.side == "B":
            # the buyer's target currency is POUNDS bought at 1/rate, so the
            # reach is set by the LOWEST rate, not the highest
            a_bot = max(params.a0 - (a_halfwidth_sds + zmax) * params.sd, 0.2)
            d_need = max(spec.L / a_bot, 1.25 * spec.T)
        else:
            d_need = max(spec.L * a_top, 1.25 * spec.T)   # worst reachable dollars
        self.jT = max(1, int(math.floor((n_d - 1) * spec.T / d_need))) #find the index which landes on T for the kink
        self.dd = spec.T / self.jT #sets the spacing so T lands exactly on node jT. 
        self.n_d = n_d
        self.d = self.dd * np.arange(n_d)#builds grid
        
        # ---- the anchor ------------------------------------------------------
        self.n_a = n_a
        self.a = params.a0 + params.sd * np.linspace(-a_halfwidth_sds, a_halfwidth_sds, n_a)

    # ---- belief interval (lo, hi) -> positions and probabilities -------------
    # A belief interval is stored as offer-price indices: lo = floor index (or
    # NO_FLOOR if none learned yet), hi = ceiling index (or n_offer if none).


    def qlo(self, lo):
        # quadrature position of the floor (0 = the very bottom, if no floor)
        return 0 if lo == NO_FLOOR else int(self.offer_boundary[lo])

    def qhi(self, hi):
        # quadrature position of the ceiling (the very top, if no ceiling)
        return self.nq if hi == self.n_offer else int(self.offer_boundary[hi])

    def PhiL(self, lo):
        # cumulative probability up to the floor (0 if no floor)
        return 0.0 if lo == NO_FLOOR else float(self.Phi[lo])

    def PhiH(self, hi):
        # cumulative probability up to the ceiling (1 if no ceiling)
        return 1.0 if hi == self.n_offer else float(self.Phi[hi])

    # ---- interpolation: reading the value tables between grid points ---------

    def interp_d(self, table, dvals):
        # Linear interpolation along the DOLLAR axis. Given target dollar values,
        # find the surrounding grid nodes and blend. 

        x = np.asarray(dvals) / self.dd              # target as a grid coordinate
        i0 = np.clip(np.floor(x).astype(np.int64), 0, self.n_d - 2)  # lower node
        frac = x - i0                                # how far toward the next node
        return table[..., i0] * (1.0 - frac) + table[..., i0 + 1] * frac 

    def matrix_at(self, W, a_val):
        # Linear interpolation along the ANCHOR axis. W is the 3-D value table
        # (anchor x pounds x dollars); this returns the 2-D (pounds, dollars)
        
        x = (a_val - self.a[0]) / (self.a[1] - self.a[0])
        x = min(max(x, 0.0), float(self.n_a - 1))
        i0 = min(int(x), self.n_a - 2)
        frac = x - i0
        return W[i0] * (1.0 - frac) + W[i0 + 1] * frac

    def bilinear(self, matrix, cvals, dvals):
        # Interpolate one (pounds, dollars) table at arbitrary (pounds, dollars)
        # used by the replay, where the trader's real position is off the grid. 
        # "Bilinear" = linear in both axes at once: find the 2x2 box
        # of surrounding nodes and blend by the fractional position in each axis.
        cx = np.asarray(cvals, dtype=float) / self.dc
        ci = np.clip(np.floor(cx).astype(np.int64), 0, self.n_c - 2)
        cf = cx - ci                                 # fractional position in pounds
        dx = np.asarray(dvals, dtype=float) / self.dd
        dj = np.clip(np.floor(dx).astype(np.int64), 0, self.n_d - 2)
        df = dx - dj                                 # fractional position in dollars
        return ((1 - cf) * ((1 - df) * matrix[ci, dj] + df * matrix[ci, dj + 1])
                + cf * ((1 - df) * matrix[ci + 1, dj] + df * matrix[ci + 1, dj + 1]))

    # ---- (c, d) evaluation where each c-ROW carries its own d-targets ---------
    # Specialised bilinear for the sell-size sweep. For a fixed sell-fraction f,
    # selling from every state (c_i, d_j) lands at c = c_i(1-f) (one value per
    # pound-row) and d = d_j + rate*f*c_i (a different dollar target per cell).
    # This evaluates the table at that whole pattern in one vectorised shot:
    # interpolate in pounds per row, then in dollars per cell (extrapolating
    # above the top dollar node). It's the hot path, so it avoids Python loops.
    def bilinear_grid(self, matrix, lc, ld):
        cx = np.asarray(lc, dtype=float) / self.dc                 # (n_c,)
        ci = np.clip(np.floor(cx).astype(np.int64), 0, self.n_c - 2)
        cf = (cx - ci)[:, None]                                    # (n_c, 1)
        row = matrix[ci] * (1 - cf) + matrix[ci + 1] * cf          # pound-interp
        dx = np.asarray(ld, dtype=float) / self.dd                 # (n_c, n_d)
        dj = np.clip(np.floor(dx).astype(np.int64), 0, self.n_d - 2)
        df = dx - dj
        lo = np.take_along_axis(row, dj, axis=1)                   # dollar-interp
        hi = np.take_along_axis(row, dj + 1, axis=1)
        return lo * (1 - df) + hi * df


# ==============================================================================
# 3. THE TERMINAL LAYER
# ==============================================================================
#
# The official accounting, straight from the rules. The team ends holding c
# pounds and d dollars; the settlement rate a5 is drawn after round 4:
#
#     W  =  c * (1 - A)  +  [ d - B * max(0, T - d) ] / a5        (pounds)
#     P/L = (W - L) / L
#
# The dollar part is what changes everything: its slope in d is (1 + B)/a5
# below the target and 1/a5 above -- a concave kink at d = T. A marginal
# dollar is worth B percent more when you are short than when you are safe,
# so the trader should pay for certainty below the target and gamble above
# it. That single kink is why d joins the state and why v2's policy is not
# v1 played four times.
#
# Because a5 is unknown when (c, d) are decided, the DP needs
# kappa(a4) = E[1/a5 | a4 = a4], the expectation of 1/a5 over the settlement
# draw a5 ~ N(a4, sd^2). Jensen (1/x convex) puts kappa(a) slightly ABOVE 1/a.
#
# On the choice of integrator. Everywhere ELSE the solver deliberately shares
# one fixed, offer-aligned quadrature grid, because the belief-bracket and
# prefix-sum machinery need the nodes aligned. kappa is the ONE expectation
# that is only ever consumed (never fed back into that aligned grid), so it is
# free to use the better tool: adaptive Gaussian quadrature (scipy.quad), which
# is purpose-built for a smooth integrand against a normal density and reaches
# machine precision, versus ~1e-6 for the shared 33-node grid. 

def kappa(grids, a4, sd):
    """E[1/a5 | a4], with a5 ~ N(a4, sd^2) truncated to a5 > 0."""
  
    lo = max(1e-9, a4 - 8.0 * sd)
    hi = a4 + 8.0 * sd
    
    num, err = quad(lambda a5: norm.pdf(a5, a4, sd) / a5, lo, hi,
                    epsabs=1e-13, epsrel=1e-13, limit=100)
    mass = norm.cdf(hi, a4, sd) - norm.cdf(lo, a4, sd)   # positive-support
    val = num / mass
    if not np.isfinite(val):
        raise ValueError
    return float(val)
  


def terminal_matrix(spec, grids, a4):
    """The settlement value table -- where backward induction starts.

    Returns a (pounds x dollars) table: the expected final wealth of FINISHING
    the game holding each (c, d), before the settlement rate is drawn. This is
    the base case W_5 that the round recursion builds up from.
    """
    if spec.terminal_mode == "dollars":
        # Gate-1 mode: score in dollars only (value = dollars, no penalties, no
        # conversion). Make a table where every pound-row is the dollar values.
        return np.tile(grids.d, (grids.n_c, 1)).astype(float)

    # Real scoring, = c(1-A) + [d - B*max(0, T-d)] * kappa :

    # seller: dollars convert at 1/a5, convex -> kappa by quadrature (Jensen).
    # buyer: pounds convert at a5, LINEAR -> E[a5 | a4] = a4 exactly.
    k = kappa(grids, a4, spec.params.sd) if spec.side == "A" else a4  #
    dpart = (grids.d - spec.B * np.maximum(spec.T - grids.d, 0.0)) * k
    cpart = grids.c * (1.0 - spec.A)
    # combine into a full (pounds x dollars) table
    return cpart[:, None] + dpart[None, :]


def terminal_wealth(spec, c, d, a5):
    """The realised accounting for one finished game (the simulator uses it)."""
    if spec.terminal_mode == "dollars":
        return d
    if spec.side == "B":
        return c * (1.0 - spec.A) + (d - spec.B * max(spec.T - d, 0.0)) * a5
    return c * (1.0 - spec.A) + (d - spec.B * max(spec.T - d, 0.0)) / a5

# ==============================================================================
# 4. THE ROUND SOLVER
# ==============================================================================
#
# One trading round entered with anchor a and continuation W_next(c, d, a').
# Two stages, solved backwards:
#
#   BdC stage (X revealed).  R(t, c, d) = max over dumps m of
#       W_next(c - m, d + (1-f) X_t m, X_t),  X_t the t-th quadrature point.
#   Cumulative sums of rate_weights * R make U_0(bracket, c, d) -- the value of walking
#   into the BdC with any belief bracket -- an O(1) prefix-sum lookup.
#
#   Offer stage (k offers left, bracket (lo, hi)).  Choosing price z_p and
#   size q = s * dc:
#
#     U_k(lo, hi) = max over (p, s), and over stopping, of
#         wA * U_{k-1}(p, hi)(c - q, d + P q)  +  wR * U_{k-1}(lo, p)(c, d)
#     wA = (Phi_hi - Phi_p) / (Phi_hi - Phi_lo),   wR = 1 - wA.
#
#   p = lo (a price already known to be below X) is a SURE sale: wA = 1.
#   q = 0 is a pure probe: no pounds move, but the bracket still tightens --
#   the "tiny trades to find the true rate" play, capped only by K.

def solve_round(spec, grids, Wnext, a):
    """Solve ONE trading round by backward induction.

    Given `Wnext` (the value of ENTERING the next round, as a table over
    (anchor, pounds, dollars)) and this round's anchor `a`, compute the value of
    entering THIS round. This is one step of the outer (across-rounds) backward
    induction; inside it runs the inner (across-offers) backward induction.

    Returns (entry_matrix, Ulevels, S0):
       entry_matrix (n_c, n_d)   value of entering the round, per (pounds, $)
       Ulevels[k]                value with k offers left, for every belief
       S0                        prefix sums behind the k=0 layer (replay reuses)
    """
    p = spec.params
    f = p.bdc_fee
    nc, nd, no, nq = grids.n_c, grids.n_d, grids.n_offer, grids.nq
    aon = spec.all_or_nothing

    # ======================================================================
    # STAGE 1: the Bureau de Change (what happens once the offers run out).
    # ======================================================================
    # R[t] = best value achievable at the BdC IF the revealed rate is sample t.
    # For each possible rate we choose how much to dump (a continuous fraction
    # of holdings; dumping 0 = carrying everything to the next round is allowed)
    # and value where we land against next round's value Wnext.
    R = np.empty((nq, nc, nd))
    for t in range(nq):
        X = _rate_at(spec, a, grids.rate_samples[t])  # this rate-sample
        Wt = grids.matrix_at(Wnext, X)               # next round's value at X
        rate = bdc_payoff_per_unit(X, f, spec.side)  # target per unit at the BdC
        best = np.full((nc, nd), -np.inf)
        for frac in grids.f_all:                     # try every dump fraction
            lc = grids.c * (1.0 - frac)        # initial ccy left after dumping
            ld = grids.d[None, :] + rate * (frac * grids.c[:, None])  # target gained
            np.maximum(best, grids.bilinear_grid(Wt, lc, ld), out=best)
        R[t] = best

    # Prefix sums of (probability * R) over the rate samples. 
    S0 = np.concatenate([np.zeros((1, nc, nd)), np.cumsum(grids.rate_weights[:, None, None] * R, axis=0)])

    # u0(lo, hi) = expected BdC value given the rate is believed to lie in the
    # interval (lo, hi). Numerator = summed value over that interval (from the
    # prefix sums); denominator = probability of that interval (renormalises,
    # since we've conditioned on the rate being inside it).
    def u0(lo, hi):
        ql, qh = grids.qlo(lo), grids.qhi(hi)
        return (S0[qh] - S0[ql]) / max(grids.cumulative_mass[qh] - grids.cumulative_mass[ql], 1e-300)

    # U0 = value with ZERO offers left, for every belief interval (lo, hi).
    # This is the base case of the inner backward induction ("no offers -> BdC").
    U0 = np.full((no + 1, no + 1, nc, nd), np.nan)
    for lo in range(-1, no):
        for hi in range(lo + 1, no + 1):
            U0[lo + 1, hi] = u0(lo, hi)

    # ======================================================================
    # STAGE 2: the offer ladder -- value with 1, 2, ... K offers left.
    # Each level is built from the one below it (backward induction).
    # ======================================================================
    Ulevels = [U0]
    for k in range(1, spec.K + 1):
        Uprev = Ulevels[-1]# value with k-1 offers left

        # --- Precompute the ACCEPT branch, already maximised over sell-size ---
        # AV[pi, hi] = best value if an offer at price pi (ceiling hi) is ACCEPTED. 
        AV = np.full((no, no + 1, nc, nd), np.nan)
        for pi in range(no):
            P = trade_rate(_rate_at(spec, a, grids.z[pi]), spec.side)  # at pi
            for hi in range(pi + 1, no + 1):
                if aon:
                    # All-or-nothing (Gate 1 only): q is forced to ALL of c, so
                    # after accepting c = 0, where every level equals U_0
                    # (nothing left to sell). So the level-0 row is exact here.
                    row0 = U0[pi + 1, hi][0]
                    matrices = np.empty((nc, nd))
                    for ci in range(nc):
                        matrices[ci] = grids.interp_d(
                            row0, grids.d + P * (ci * grids.dc))
                    AV[pi, hi] = matrices
                else:
                    # Continuous sizing: try every sell-fraction, land at
                    # (c(1-frac), d + P*frac*c), value against the k-1 level.
                    child = Uprev[pi + 1, hi]
                    best = np.full((nc, nd), -np.inf)
                    F = grids.f_all if spec.q_floor < 0 else grids.f_pos
                    for frac in F:
                        lc = grids.c * (1.0 - frac)
                        ld = grids.d[None, :] + P * (frac * grids.c[:, None])
                        np.maximum(best, grids.bilinear_grid(child, lc, ld),
                                   out=best)
                    AV[pi, hi] = best

        # --- Combine into U_k: for each belief, pick the best offer ---------
        Unew = np.full_like(U0, np.nan)
        for lo in range(-1, no):
            if aon and lo != NO_FLOOR:
                continue          # floors never arise under all-or-nothing
            Pl = grids.PhiL(lo)
            for hi in range(lo + 1, no + 1):
                mass = grids.PhiH(hi) - Pl           # probability of (lo, hi)
                best = U0[lo + 1, hi].copy()         # option: stop, take the BdC
                p_start = 0 if lo == NO_FLOOR else lo  # p = lo is the "sure sale"
                for pi in range(p_start, hi):        # try each candidate price
                    # accept/reject probabilities given the belief (lo, hi)
                    wA = (grids.PhiH(hi) - grids.Phi[pi]) / mass
                    wR = (grids.Phi[pi] - Pl) / mass
                    # value = P(accept)*accept-value + P(reject)*continue-value.
                    # Reject continues at the SAME (c,d) with a tightened ceiling.
                    val = wA * AV[pi, hi]
                    if wR > 1e-15:
                        val = val + wR * Uprev[lo + 1, pi]
                    np.maximum(best, val, out=best)
                Unew[lo + 1, hi] = best
        Ulevels.append(Unew)

    # Value of entering the round = K offers left, belief wide open (no floor,
    # no ceiling) -- the trader knows nothing yet.
    entry = Ulevels[spec.K][NO_FLOOR + 1, no]
    return entry, Ulevels, S0


# ==============================================================================
# 5. THE FULL SOLVER
# ==============================================================================

class V2Solution:
    """spec, grids; Wtables[n] = value of entering round n on the anchor grid
    (shape (n_a, n_c, n_d); Wtables[rounds+1] is the terminal layer)
    
    entry1 is value of entering round 1 for every (pounds, dollars) position; 
    store holds the tables the replay acts from
    
    value is the headline E[W] from (L, 0, a0)."""

    def __init__(self, spec, grids, Wtables, entry1, store):
        self.spec, self.grids = spec, grids
        self.Wtables, self.entry1, self.store = Wtables, entry1, store
        self.value = float(entry1[grids.n_c - 1, 0])

    #Proit-loss
    def pl(self):
        return (self.value - self.spec.L) / self.spec.L

#Bundle one solved round's tables for later use by the replay
def _pack(spec, Ulevels, S0):
    out = {"S0": S0.astype(np.float32),
           "U": {k: Ulevels[k].astype(np.float32) for k in range(1, spec.K)}}
    return out


def solve_v2(spec, grids=None, print_progress=False, store_tables=True):
    """Solve the whole game: chain the rounds by backward induction.

    Seeds the recursion with the settlement layer, then solves round R, R-1,
    ..., 2 (each using the already-solved NEXT round as its reward), and finally
    round 1 at the single known starting rate. Returns a V2Solution whose
    `.value` is the headline number: best expected final wealth from the start.
    """
    grids = grids or Grids(spec)
    # base case: the settlement value table at every anchor node
    Wterm = np.stack([terminal_matrix(spec, grids, a4) for a4 in grids.a])
    Wtables = {spec.rounds + 1: Wterm}     # value of "entering" the settlement
    store = {}                           # per-round tables the replay will use
    Wnext = Wterm

    # walk backwards: round R down to round 2. Round n's value is built from
    # round n+1's value (Wnext), solved at every possible anchor.
    for n in range(spec.rounds, 1, -1):
        t0 = time.time()
        Wn = np.empty_like(Wnext)
        for ia, a in enumerate(grids.a):             # solve at each anchor node
            entry, Ulv, S0 = solve_round(spec, grids, Wnext, a)
            Wn[ia] = entry                           # value of entering round n
            if store_tables:
                store[(n, ia)] = _pack(spec, Ulv, S0)
        Wtables[n] = Wn
        Wnext = Wn                                   # becomes next round's reward
        if print_progress:
            print(f"  round {n}: {grids.n_a} anchor nodes solved in "
                  f"{time.time() - t0:.1f}s")

    # round 1 is special: the game always starts at the known rate a0, so we
    # solve it only there (not at every anchor).
    entry1, Ulv1, S01 = solve_round(spec, grids, Wnext, spec.params.a0)
    if store_tables:
        store[(1, None)] = _pack(spec, Ulv1, S01)
    return V2Solution(spec, grids, Wtables, entry1, store)


# ==============================================================================
# 6. ACTING FROM THE TABLES
# ==============================================================================
#
# The replay does NOT read a stored argmax: it re-derives the greedy action
# from the value tables at the trader's actual (possibly off-grid) state,
# interpolating bilinearly in (c, d) and blending action values between the
# two anchor nodes that bracket the true anchor. Prices are placed at
# P = a + sd * z_p with the TRUE anchor a: the z-frame is what the tables
# share across anchors.
 
def _tabs_for(sol, n, a):
    #The stored tables (with blend weights) for round n at true anchor a.
    if n == 1:
        return [(sol.store[(1, None)], 1.0)]
    grids = sol.grids
    x = (a - grids.a[0]) / (grids.a[1] - grids.a[0])
    x = min(max(x, 0.0), float(grids.n_a - 1))
    i0 = min(int(x), grids.n_a - 2)
    frac = x - i0
    return [(sol.store[(n, i0)], 1.0 - frac), (sol.store[(n, i0 + 1)], frac)]
 
 
def _u0_point(grids, tab, lo, hi, c, d):
    #U_0(lo, hi) at one real (c, d) point, from the stored prefix sums.
    S0 = tab["S0"]
    ql, qh = grids.qlo(lo), grids.qhi(hi)
    v = grids.bilinear(S0[qh] - S0[ql], np.array([c]), np.array([d]))[0]
    return v / max(grids.cumulative_mass[qh] - grids.cumulative_mass[ql], 1e-300)
 
 
def greedy_offer_action(sol, n, k_rem, lo, hi, c, d, a):
    """Best (price index, size) for the state, or None to stop and take the
    BdC. Mirrors the DP's own action set: grid prices, sizes on c-grid steps
    plus 'everything'."""
    grids, spec = sol.grids, sol.spec
    sd = spec.params.sd
    tabs = _tabs_for(sol, n, a)
 
    stop = sum(w * _u0_point(grids, tab, lo, hi, c, d) for tab, w in tabs)
 
    p_start = 0 if lo == NO_FLOOR else lo
    if p_start >= hi or c < 0:
        return None, stop
    price_indices = np.arange(p_start, hi)               # candidate offer prices
    price_values = trade_rate(_rate_at(spec, a, grids.z[price_indices]),
                              spec.side)     # in the trader's units (target/unit)
    PhiH, PhiL = grids.PhiH(hi), grids.PhiL(lo)
    mass = PhiH - PhiL
    wA = (PhiH - grids.Phi[price_indices]) / mass         # P(accept) per price
    wR = (grids.Phi[price_indices] - PhiL) / mass         # P(reject) per price
 
    if spec.all_or_nothing:
        size_options = np.array([c])
    else:
        # same continuous action set as the DP: q = f * c
        if c <= 1e-9:
            return None, stop
        F = grids.f_all if spec.q_floor < 0 else grids.f_pos
        size_options = F * c
    n_sizes, n_prices = size_options.size, price_indices.size
 
    # ---- bilinear interpolation setup for the accept branch --------------
    # Accepting a size-q offer at price P lands at (c - q, d + P*q) -- which
    # varies with BOTH the size and the price, so we set up the surrounding
    # grid-node indices and fractional positions for every (size, price) pair.
    pounds_after = c - size_options                       
    pound_coord = pounds_after / grids.dc                 
    pound_lo = np.clip(np.floor(pound_coord).astype(np.int64), 0, grids.n_c - 2)
    pound_frac = (pound_coord - pound_lo)[:, None]        # how far to next pound node
    dollars_after = d + price_values[None, :] * size_options[:, None]  # $ if accepted
    dollar_coord = dollars_after / grids.dd
    dollar_lo = np.clip(np.floor(dollar_coord).astype(np.int64), 0, grids.n_d - 2)
    dollar_frac = dollar_coord - dollar_lo               # how far to next dollar node
    pound_lo_col = pound_lo[:, None]
    price_col = np.arange(n_prices)[None, :]
 
    total = np.zeros((n_sizes, n_prices))
    for tab, wgt in tabs:
        if wgt == 0.0:
            continue
        # ---- accept branch: child belief (price, hi), level k_rem - 1 -------
        if k_rem - 1 >= 1:
            # continuation is the offer-ladder table one level down
            matrices = tab["U"][k_rem - 1][price_indices + 1, hi]  # (n_prices,nc,nd)
            v00 = matrices[price_col, pound_lo_col, dollar_lo]
            v01 = matrices[price_col, pound_lo_col, dollar_lo + 1]
            v10 = matrices[price_col, pound_lo_col + 1, dollar_lo]
            v11 = matrices[price_col, pound_lo_col + 1, dollar_lo + 1]
            accept_val = (1 - pound_frac) * ((1 - dollar_frac) * v00
                                             + dollar_frac * v01) \
                + pound_frac * ((1 - dollar_frac) * v10 + dollar_frac * v11)
        else:#Last offer
            # accepting uses the last offer -> continuation is the BdC (via the
            # prefix sums), evaluated over the tightened belief (price, hi)
            S0 = tab["S0"]
            q_ceiling = grids.qhi(hi)
            q_floor_price = grids.offer_boundary[price_indices]
            bdc_interval = S0[q_ceiling][None] - S0[q_floor_price]  # (n_prices,nc,nd)
            v00 = bdc_interval[price_col, pound_lo_col, dollar_lo]
            v01 = bdc_interval[price_col, pound_lo_col, dollar_lo + 1]
            v10 = bdc_interval[price_col, pound_lo_col + 1, dollar_lo]
            v11 = bdc_interval[price_col, pound_lo_col + 1, dollar_lo + 1]
            numerator = (1 - pound_frac) * ((1 - dollar_frac) * v00
                                            + dollar_frac * v01) \
                + pound_frac * ((1 - dollar_frac) * v10 + dollar_frac * v11)
            accept_val = numerator / np.maximum(
                grids.cumulative_mass[q_ceiling]
                - grids.cumulative_mass[q_floor_price], 1e-300)[None, :]
        # ---- reject branch: child belief (lo, price) at unchanged (c, d) -----
        pound_coord0 = min(max(c / grids.dc, 0.0), grids.n_c - 1.0)
        pound_lo0 = min(int(pound_coord0), grids.n_c - 2)
        pound_frac0 = pound_coord0 - pound_lo0
        dollar_coord0 = d / grids.dd
        dollar_lo0 = min(max(int(dollar_coord0), 0), grids.n_d - 2)
        dollar_frac0 = dollar_coord0 - dollar_lo0

        # Bilinear-interpolate `stack` at the trader's current (c, d)
        def _at_point(stack):                # stack (n_prices,nc,nd) -> (n_prices,)
            return ((1 - pound_frac0) * ((1 - dollar_frac0) * stack[:, pound_lo0, dollar_lo0]
                                         + dollar_frac0 * stack[:, pound_lo0, dollar_lo0 + 1])
                    + pound_frac0 * ((1 - dollar_frac0) * stack[:, pound_lo0 + 1, dollar_lo0]
                                     + dollar_frac0 * stack[:, pound_lo0 + 1, dollar_lo0 + 1]))
 
        if k_rem - 1 >= 1:
            reject_val = _at_point(tab["U"][k_rem - 1][lo + 1, price_indices])
        else:
            S0 = tab["S0"]
            q_floor = grids.qlo(lo)
            reject_val = _at_point(
                S0[grids.offer_boundary[price_indices]] - S0[q_floor][None])
            reject_val = reject_val / np.maximum(
                grids.cumulative_mass[grids.offer_boundary[price_indices]]
                - grids.cumulative_mass[q_floor], 1e-300)
        reject_val = np.where(wR > 1e-15, reject_val, 0.0)
        total += wgt * (wA[None, :] * accept_val + (wR * reject_val)[None, :])
 
    best_flat = int(np.argmax(total))
    size_i, price_i = divmod(best_flat, n_prices)
    if total[size_i, price_i] <= stop + 1e-12:
        return None, stop
    return (int(price_indices[price_i]), float(size_options[size_i])), \
        float(total[size_i, price_i])
 
 
def greedy_bdc_action(sol, n, c, d, X):
    """Best BdC dump m at the revealed rate X, from the continuation table.
    The action set is the DP's own: a continuous fraction of holdings (carry
    included)."""
    grids, spec = sol.grids, sol.spec
    matrix = grids.matrix_at(sol.Wtables[n + 1], X)
    m = grids.f_all * c
    vals = grids.bilinear(matrix, c - m,
                      d + bdc_payoff_per_unit(X, spec.params.bdc_fee,
                                               spec.side) * m)
    return float(m[int(np.argmax(vals))])
 
 
# ==============================================================================
# 6.5 TRACING AND VISUALISING OPTIMAL PLAY
# ==============================================================================
#
# The value tables answer "how well can you do"; this section answers "what
# does the optimal player actually DO". Optimal play is a CONTINGENT strategy,
# not a fixed script -- the second offer depends on whether the first was
# accepted, and the next round depends on this round's realised path -- so the
# honest artifact is a traced PLAYTHROUGH: fix a rate path, let the greedy
# policy (identical to the replay's) play it, and record every (P, q) offer,
# each accept/reject, the BdC dump, and the running position (c, d).
#
# play_game returns that trace; print_game_trace prints it as a per-offer log;
# plot_game draws it as a two-panel timeline (the offer ladder against the
# hidden rate, and inventory draining toward the target). All three are driven
# by exactly the actions the DP would take, through the shared game rules.
 
def play_game(sol, seed=None, rng=None, path=None, a5=None):
    """Play one game under the optimal policy and return a full trace.
 
    path : optional array [X_1..X_R] to force a specific realised rate path
           (so a chosen percentile game can be replayed); a5 forces the
           settlement rate. Otherwise both are drawn from rng/seed.
    """
    spec, grids = sol.spec, sol.grids
    p = spec.params
    if rng is None:
        rng = np.random.default_rng(seed)
    c, d, a = spec.L, 0.0, p.a0
    rounds = []
    for n in range(1, spec.rounds + 1):
        X = float(path[n - 1]) if path is not None else a + p.sd * rng.standard_normal()
        lo, hi = NO_FLOOR, grids.n_offer
        offers = []
        for k_rem in range(spec.K, 0, -1):
            action, _ = greedy_offer_action(sol, n, k_rem, lo, hi, c, d, a)
            if action is None:
                break                                   # stop early: to the BdC
            pi, q = action
            price = _rate_at(spec, a, grids.z[pi])
            P = trade_rate(price, spec.side)
            accepted = mm_accepts(price, X, spec.side)                  
            if accepted:
                c -= q
                d += P * q
                lo = pi                                  # floor learned
            else:
                hi = pi                                  # ceiling learned
            # "P" is the trader's rate (target per unit held); "quote" is the
            # $/GBP number on the slip. They differ for the buyer, and only the
            # quote is comparable with the hidden rate X -- so plots that draw
            # an offer against X must use the quote, not P.
            offers.append({"offer_no": spec.K - k_rem + 1, "z": grids.z[pi],
                           "P": P, "quote": price, "q": q, "accepted": accepted,
                           "c": c, "d": d})
        m = greedy_bdc_action(sol, n, c, d, X)           # BdC sub-phase
        if m > 0:
            d += bdc_payoff_per_unit(X, p.bdc_fee, spec.side) * m
            c -= m
        rounds.append({"n": n, "anchor": a, "X": X, "offers": offers,
                       "bdc_dump": m, "c": c, "d": d})
        a = X
    if a5 is None:
        a5 = a + p.sd * rng.standard_normal()
    W = terminal_wealth(spec, c, d, a5)
    return {"spec": spec, "rounds": rounds, "a5": float(a5),
            "c": c, "d": d, "W": float(W), "pl": (W - spec.L) / spec.L}
 
 
def print_game_trace(trace):
    """The per-offer log: round by round, every (P, q) and its verdict, the
    BdC dump, and the running position -- the literal 'how the optimal player
    plays' table."""
    spec = trace["spec"]
    i_sym, t_sym = ("GBP", "$") if spec.side == "A" else ("$", "GBP")
    print(f"{'':2}anchor   hidden X      offer            verdict     "
          f"-> {i_sym} left   {t_sym} banked")
    for r in trace["rounds"]:
        zX = (r["X"] - r["anchor"]) / spec.params.sd
        head = f"Round {r['n']}: a={r['anchor']:.4f}  X={r['X']:.4f} (z={zX:+.2f})"
        print("-" * 78)
        print(head)
        if not r["offers"]:
            print("  (no offer made -- straight to the BdC)")
        for o in r["offers"]:
            verdict = "ACCEPT" if o["accepted"] else "reject"
            print(f"  offer {o['offer_no']}: {o['P']:.4f} "
                  f"{rate_units(spec.side)} (slip {o['quote']:.4f} $/GBP, "
                  f"z={o['z']:+.2f})  "
                  f"sell {i_sym} {o['q']:>8,.0f}   {verdict:>6}   "
                  f"-> c={o['c']:>8,.0f}   d={o['d']:>10,.0f}")
        if r["bdc_dump"] > 1e-6:
            print(f"  BdC: dump {i_sym} {r['bdc_dump']:>8,.0f} at the BdC rate"
                  f"                -> c={r['c']:>8,.0f}   d={r['d']:>10,.0f}")
        else:
            print(f"  BdC: carry (no dump)"
                  f"                             -> c={r['c']:>8,.0f}   "
                  f"d={r['d']:>10,.0f}")
    print("=" * 78)
    short = max(0.0, spec.T - trace["d"])
    print(f"Settlement a5={trace['a5']:.4f} | end c={i_sym} {trace['c']:,.0f}, "
          f"d={t_sym} {trace['d']:,.0f} (target {t_sym} {spec.T:,.0f}, "
          f"{'short %s %.0f' % (t_sym, short) if short > 1 else 'target met'}) | "
          f"W={i_sym} {trace['W']:,.0f}  P/L={trace['pl']*100:+.2f}%")
 
 
def plot_game(trace, save_path=None, ax_rate=None, ax_inv=None, title=None):
    """Two stacked panels sharing the game timeline:
       top    the offer ladder against the hidden rate, everything shown in
              the TRADER's units (target per unit held) so both sides read the
              same way -- filled green = accepted, open red = rejected, marker
              size ~ initial currency offered;
       bottom inventory: initial currency remaining (drains to 0) and target
              currency banked as a fraction of T (should cross 1.0)."""
    spec = trace["spec"]
    side = spec.side
    i_sym, t_sym = ("GBP", "$") if side == "A" else ("$", "GBP")
    own_fig = ax_rate is None
    if own_fig:
        fig, (ax_rate, ax_inv) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]})
 
    x = 0
    round_spans = []
    cser_x, cser_y, dser_x, dser_y = [0], [spec.L], [0], [0.0]
    for r in trace["rounds"]:
        x0 = x + 0.5
        n_ev = max(len(r["offers"]), 1)
        # hidden rate as a band across this round's offers
        ax_rate.hlines(trade_rate(r["X"], side), x0, x0 + n_ev,
                       color="tab:purple", lw=1.6,
                       ls="--", alpha=0.8,
                       label="hidden true rate X" if r["n"] == 1 else None)
        ax_rate.plot([x0 - 0.3], [trade_rate(r["anchor"], side)],
                     marker="_", ms=14,
                     color="grey",
                     label="anchor (last revealed)" if r["n"] == 1 else None)
        for o in r["offers"]:
            x += 1
            size = 40 + 320 * (o["q"] / spec.L)
            if o["accepted"]:
                ax_rate.scatter([x], [o["P"]], s=size, c="tab:green",
                                edgecolors="k", zorder=5,
                                label="offer accepted" if r["n"] == 1 and
                                o is r["offers"][0] else None)
            else:
                ax_rate.scatter([x], [o["P"]], s=size, facecolors="none",
                                edgecolors="tab:red", linewidths=1.6, zorder=5,
                                label="offer rejected" if r["n"] == 1 and
                                not any(oo["accepted"] for oo in
                                        r["offers"][:r["offers"].index(o)])
                                else None)
            cser_x.append(x); cser_y.append(o["c"])
            dser_x.append(x); dser_y.append(o["d"])
        if len(r["offers"]) == 0:
            x += 1
        if r["bdc_dump"] > 1e-6:
            ax_rate.scatter([x + 0.4],
                            [bdc_payoff_per_unit(r["X"], spec.params.bdc_fee,
                                                 side)],
                            marker="s", s=70, c="tab:orange", edgecolors="k",
                            zorder=5,
                            label="BdC dump" if not any(
                                rr["bdc_dump"] > 1e-6 for rr in
                                trace["rounds"][:trace["rounds"].index(r)])
                            else None)
        cser_x.append(x + 0.5); cser_y.append(r["c"])
        dser_x.append(x + 0.5); dser_y.append(r["d"])
        round_spans.append((x0 - 0.5, x + 0.7, r["n"]))
        x += 1
 
    # settlement
    ax_rate.plot([x + 0.3], [trade_rate(trace["a5"], side)], marker="*",
                 ms=16, c="k", label="settlement a5")
 
    for (lo, hi, n) in round_spans:
        ax_rate.axvspan(lo, hi, color="grey", alpha=0.05)
        ax_rate.text((lo + hi) / 2, ax_rate.get_ylim()[1], f"round {n}",
                     ha="center", va="bottom", fontsize=8, color="grey")
    ax_rate.set_ylabel(f"rate ({rate_units(side)} -- target per unit held)")
    ax_rate.set_title(title or
                      f"Optimal play, P/L = {trace['pl']*100:+.2f}%")
    ax_rate.legend(fontsize=7, loc="upper left", ncol=2)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
 
    ax_inv.step(cser_x, np.array(cser_y) / spec.L, where="post",
                color="tab:blue", lw=1.8, label=f"{i_sym} left / L")
    ax_inv.step(dser_x, np.array(dser_y) / spec.T, where="post",
                color="tab:green", lw=1.8, label=f"{t_sym} banked / T")
    ax_inv.axhline(1.0, ls=":", color="grey", label="target")
    ax_inv.set_ylabel("fraction")
    ax_inv.set_xlabel("offer / event through the game")
    ax_inv.legend(fontsize=7, loc="center left")
    ax_inv.spines["top"].set_visible(False)
    ax_inv.spines["right"].set_visible(False)
 
    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
 
 
# ==============================================================================
# 6.6 THE MAIN OUTPUT: the optimal strategy for a given trader
# ==============================================================================
#
# One call that takes a trader (TraderSpec), solves the game, and produces the 
# oprimal strategy on asingle random rate path
 
def _pick_representative_game(sol, res):
    """From replay draws `res`, pick a multi-round game near the median outcome
    (so the drawn path shows the strategy at work, not an instant round-1
    clear)."""
    med = np.median(res["W"])
    best = None
    for i in range(len(res["W"])):
        tr = play_game(sol, path=res["X"][i], a5=res["a5"][i])
        if tr["rounds"][0]["c"] > 1 and \
                sum(1 for r in tr["rounds"] if r["offers"]) >= 2:
            s = abs(tr["W"] - med)
            if best is None or s < best[0]:
                best = (s, tr)
    if best is not None:
        return best[1]
    mid = int(np.argsort(res["W"])[len(res["W"]) // 2])
    return play_game(sol, path=res["X"][mid], a5=res["a5"][mid])
 
 
def optimal_strategy(spec, rate_path=None, a5=None, seed=11,
                     grid=None, save_path="v2_optimal_strategy.png",
                     n_scan=4000, print_progress=True):
    """Solve the game for `spec` and draw the path the optimal policy follows.
 
    spec       the trader to solve for (all parameters: L, T, A, B, K, rounds).
    rate_path  optional [X_1,..,X_R] to force the revealed rates; otherwise a
               representative multi-round game is chosen automatically.
    a5         optional settlement rate (used with rate_path).
    grid       optional Grids; defaults to the standard resolution.
    Returns (solution, trace). Prints the move-by-move log and saves the figure.
    """
    if print_progress:
        print(f"Solving the game for this trader "
              f"(L={spec.L:,.0f}, T={spec.T:,.0f}, A={spec.A}, B={spec.B}, "
              f"rounds={spec.rounds}, K={spec.K}) ...", flush=True)
    sol = solve_v2(spec, grid, store_tables=True)
    if print_progress:
        print(f"  solved.  E[final wealth] = {sol.value:,.0f}  "
              f"(expected P/L {sol.pl()*100:+.2f}%)\n", flush=True)
 
    # choose the game to display
    if rate_path is not None:
        trace = play_game(sol, path=np.asarray(rate_path, dtype=float), a5=a5)
    else:
        res = simulate_paths(sol, n_paths=n_scan, seed=seed)
        trace = _pick_representative_game(sol, res)
 
    if print_progress:
        print("THE OPTIMAL STRATEGY, move by move "
              "(for this rate draw):\n")
        print_game_trace(trace)
 
    plot_game(trace, save_path=save_path,
              title=f"Optimal strategy  |  P/L = {trace['pl']*100:+.2f}%  "
                    f"(L={spec.L:,.0f}, T={spec.T:,.0f})")
    if print_progress:
        print(f"\nPath drawn to: {save_path}")
    return sol, trace
 
 # ==============================================================================
# 7. POLICY REPLAY  
# ==============================================================================
#
# Play the tables for real, through the SHARED rules -- mm_accepts decides
# every trade and bdc_payoff_per_unit prices every dump, exactly as in v0 and
# v1. Every path also keeps the books of all three parties so that money
# conservation (gate 4) is checked on the same run.


def simulate_paths(sol, n_paths=20_000, seed=0, print_progress=False):
    spec, grids = sol.spec, sol.grids
    p = spec.params
    rng = np.random.default_rng(seed)
    W = np.empty(n_paths)
    Xs = np.empty((n_paths, spec.rounds))
    a5s = np.empty(n_paths)
    max_pound_err = 0.0
    max_dollar_err = 0.0

    for ipath in range(n_paths):
        c, d = spec.L, 0.0
        mm_p = mm_d = 0.0
        bdc_p = bdc_d = 0.0
        a = p.a0
        for n in range(1, spec.rounds + 1):
            X = a + p.sd * rng.standard_normal()
            Xs[ipath, n - 1] = X
            lo, hi = NO_FLOOR, grids.n_offer
            for k_rem in range(spec.K, 0, -1):
                action, _ = greedy_offer_action(sol, n, k_rem, lo, hi, c, d, a)
                if action is None:
                    break
                p_i, q = action
                price = _rate_at(spec, a, grids.z[p_i])
                P = trade_rate(price, spec.side)
                if mm_accepts(price, X, spec.side):   # the shared rule
                    c -= q
                    d += P * q
                    mm_p += q
                    mm_d -= P * q
                    lo = p_i
                else:
                    hi = p_i
            # BdC stage: X is revealed
            m = greedy_bdc_action(sol, n, c, d, X)
            if m > 0:
                pay = bdc_payoff_per_unit(X, p.bdc_fee, spec.side)  # shared
                pay = pay * m
                c -= m
                d += pay
                bdc_p += m
                bdc_d -= pay
            a = X
        a5 = a + p.sd * rng.standard_normal()
        a5s[ipath] = a5
        W[ipath] = terminal_wealth(spec, c, d, a5)
        max_pound_err = max(max_pound_err, abs((c + mm_p + bdc_p) - spec.L))
        max_dollar_err = max(max_dollar_err, abs(d + mm_d + bdc_d))
        if print_progress and (ipath + 1) % 5000 == 0:
            print(f"    {ipath + 1}/{n_paths} paths")
    return {"W": W, "X": Xs, "a5": a5s,
            "pound_err": max_pound_err, "dollar_err": max_dollar_err}


# ==============================================================================
# 8. THE CLAIRVOYANT BENCHMARK
# ==============================================================================
#
# Hindsight optimum, per path, in closed form. With the whole rate path known,
# every pound sold goes to the MM in the best round at (just below) the peak
# rate r* = max(X_1..X_R): the BdC's (1-f) X_n and every other round are
# dominated. The only real decision left is HOW MANY pounds s to sell:
#
#     W_cv(s) = (L - s)(1 - A) + [ s r* - B max(0, T - s r*) ] / a5

def clairvoyant_wealth(spec, Xs, a5s):
    # best target-per-initial-unit in the best round: the seller wants the
    # HIGHEST rate, the buyer (paying that rate per pound) wants the LOWEST
    if spec.side == "B":
        r = (1.0 / Xs).max(axis=1)
    else:
        r = Xs.max(axis=1)
    cands = [np.zeros_like(r), np.full_like(r, spec.L)]
    s_kink = spec.T / r
    cands.append(np.where(s_kink <= spec.L, s_kink, 0.0))
    best = np.full(r.shape, -np.inf)
    for s in cands:
        if spec.side == "B":
            w = (spec.L - s) * (1 - spec.A) \
                + (s * r - spec.B * np.maximum(spec.T - s * r, 0.0)) * a5s
        else:
            w = (spec.L - s) * (1 - spec.A) \
                + (s * r - spec.B * np.maximum(spec.T - s * r, 0.0)) / a5s
        best = np.maximum(best, w)
    return best

# The FLOOR, in closed form. The mirror of the clairvoyant: sell nothing to the
# MM and dump the whole book at the BdC in round 1, then hold dollars to
# settlement. No skill, no information used -- what the trader gets for free.
#
#     d(a1) = L (1-f) a1 ,  c = 0
#     W     = [ d - B max(0, T - d) ] / a5
#
# a1 is round 1's revealed rate; settlement is `rounds` FURTHER random-walk
# steps away, so a5 | a1 ~ N(a1, rounds*sd^2) -- kappa takes sqrt(rounds)*sd,
# NOT sd. The outer expectation over a1 reuses the solver's own quadrature.

def bdc_baseline(spec, grids=None):
    """E[W] and P/L of always-BdC: the do-nothing floor the DP must beat."""
    grids = grids or Grids(spec)
    p = spec.params
    sd_to_settlement = math.sqrt(spec.rounds) * p.sd
    EW = 0.0
    for z, w in zip(grids.rate_samples, grids.rate_weights):
        a1 = p.a0 + p.sd * z
        d = spec.L * bdc_payoff_per_unit(a1, p.bdc_fee, spec.side)
        payoff = d - spec.B * max(spec.T - d, 0.0)
        # buyer settles LINEARLY, and the walk is a martingale: E[a5 | a1] = a1
        EW += w * payoff * (kappa(grids, a1, sd_to_settlement)
                            if spec.side == "A" else a1)
    return EW, (EW - spec.L) / spec.L
# ==============================================================================
# 9. VALIDATION
# ==============================================================================

def gate1_v1_anchor(K=3, print_progress=True):
    """One round, all-or-nothing sizes, expected-DOLLAR objective: v2's
    machinery must collapse to v1's solver, which is itself anchored to v0's
    closed form. Run at a fine price grid; compare every offer budget up to K.
    """
    from v1_sequential_offers import solve_v1
    from v0_game import closed_form_optimal_rate, expected_target_per_unit

    # side is PINNED to "A" here: v1 and v0's closed form model the seller,
    # so this spec must not inherit the file's default side. Without the pin,
    # a default of side="B" makes gate 1 compare a buyer-v2 against a
    # seller-v1 and fail (or worse, pass for the wrong reason).
    spec = TraderSpec(rounds=1, K=K, B=0.0, all_or_nothing=True,
                      terminal_mode="dollars", side="A")
    grids = Grids(spec, n_offer=161, quad_per_cell=3, n_c=2, n_d=5, n_a=3)
    _, Ulv, _ = solve_round(spec, grids, np.stack(
        [terminal_matrix(spec, grids, a4) for a4 in grids.a]), spec.params.a0)

    v1 = solve_v1(max_offers=K)
    worst = 0.0
    for k in range(1, K + 1):
        v2_val = Ulv[k][NO_FLOOR + 1, grids.n_offer][grids.n_c - 1, 0] / spec.L
        gap = abs(v2_val - v1.value(k))
        worst = max(worst, gap)
        if print_progress:
            print(f"  K={k}: v2 {v2_val:.6f}  v1 {v1.value(k):.6f}  "
                  f"gap {gap:.2e}")
    P_star = closed_form_optimal_rate(A0_DEFAULT, sd_DEFAULT, BDC_FEE_DEFAULT)
    v0_val = expected_target_per_unit(P_star)
    gap0 = abs(Ulv[1][NO_FLOOR + 1, grids.n_offer][grids.n_c - 1, 0] / spec.L - v0_val)
    if print_progress:
        print(f"  K=1 vs v0 closed form: gap {gap0:.2e}")
    return worst

def gate3_and_4(sol, n_paths=20_000, seed=1, print_progress=True):
    """Gate 3: a Monte Carlo of the tables, played through the shared rules,
    must land on the DP's claimed value. Gate 4: the three parties' books must
    conserve every pound and every dollar on every path."""
    res = simulate_paths(sol, n_paths=n_paths, seed=seed)
    mc = float(res["W"].mean())
    se = float(res["W"].std(ddof=1) / math.sqrt(n_paths))
    gap = mc - sol.value
    tol = max(5 * se, 2e-3 * sol.spec.L)
    if print_progress:
        print(f"  DP claim {sol.value:,.1f}   MC {mc:,.1f} +/- {2*se:,.1f} "
              f"(2 s.e.)   gap {gap:+,.1f}  [tolerance {tol:,.1f}]")
        i_sym, t_sym = ("GBP", "$") if sol.spec.side == "A" else ("$", "GBP")
        print(f"  money conservation: worst {i_sym} error "
              f"{res['pound_err']:.2e}, worst {t_sym} error "
              f"{res['dollar_err']:.2e}")
    assert abs(gap) < tol, f"gate 3 FAILED: gap {gap:+.1f} vs tol {tol:.1f}"
    assert res["pound_err"] < 1e-6 * sol.spec.L, "gate 4 FAILED (pounds)"
    assert res["dollar_err"] < 1e-6 * sol.spec.L, "gate 4 FAILED (dollars)"
    return res

def gate5_brackets(sol, res, print_progress=True):
    """The DP must beat the do-nothing floor and lose to hindsight."""
    spec = sol.spec
    W_bdc, pl_bdc = bdc_baseline(spec, sol.grids)
    W_cv = float(clairvoyant_wealth(spec, res["X"], res["a5"]).mean())
    if print_progress:
        i_sym = "GBP" if spec.side == "A" else "$"
        print(f"  always-BdC floor    {i_sym} {W_bdc:,.1f}  ({pl_bdc*100:+.3f}%)")
        print(f"  DP                  {i_sym} {sol.value:,.1f}  ({sol.pl()*100:+.3f}%)")
        print(f"  clairvoyant ceiling {i_sym} {W_cv:,.1f}  ({(W_cv-spec.L)/spec.L*100:+.3f}%)")
    assert sol.value > W_bdc, "gate 5 FAILED: DP is worse than doing nothing"
    assert sol.value < W_cv,  "gate 5 FAILED: DP beats hindsight"
    return W_bdc, W_cv

def terminal_examples():
    """The two worked examples in the game rules, as unit tests."""
    spec = TraderSpec(L=100_000.0, T=125_000.0, A=0.02, B=0.03, side = "A")
    # residue: GBP 1,000 left over costs A% of it, i.e. GBP 20
    w = terminal_wealth(spec, 1000.0, spec.T, 1.30)
    assert abs(w - (980.0 + spec.T / 1.30)) < 1e-9
    # deficit: USD 1,000 short of target costs B% of it, converted at a5
    w = terminal_wealth(spec, 0.0, spec.T - 1000.0, 1.30)
    assert abs(w - ((spec.T - 1000.0) - 0.03 * 1000.0) / 1.30) < 1e-9
    print("  both worked examples from the rules: PASSED (seller's card)")
    # the rules print worked examples for the seller only; these are the same
    # two rules applied to the buyer's card, so both sides' accounting is tested
    specB = TraderSpec(L=500_000.0, T=396_000.0, A=0.00, B=0.20, side="B")
    # residue: T4 carries no penalty on leftover dollars
    w = terminal_wealth(specB, 10_000.0, specB.T, 1.30)
    assert abs(w - (10_000.0 + specB.T * 1.30)) < 1e-9
    # deficit: GBP 1,000 short of target costs B% of it, converted at a5
    w = terminal_wealth(specB, 0.0, specB.T - 1000.0, 1.30)
    assert abs(w - ((specB.T - 1000.0) - 0.20 * 1000.0) * 1.30) < 1e-9
    print("  the same two rules on the buyer's card: PASSED")

# ==============================================================================
# 10. PLOTS
# ==============================================================================
def plot_kink_and_policy(sol, save_path="v2_fig1_policy.png"):
    """Left: the marginal value, in initial currency, of one more unit of
    TARGET currency banked, by round -- the
    settlement kink at the target coming into focus as the deadline nears.
    Right: the last-offer price by round, showing the deadline (not the kink)
    driving aggression. Both are read from the solved tables at a fixed
    reference state (full book, open belief, anchor a0), so only the round
    varies -- this isolates the mechanism, it is not an average over played
    games."""
    spec, grids = sol.spec, sol.grids
    i_sym, t_sym = ("GBP", "$") if spec.side == "A" else ("$", "GBP")
    ia0 = (grids.n_a - 1) // 2                     # the a0 node
    ci = (grids.n_c - 1) // 2                      # a mid-book row (c = L/2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- left: d-slope of the value entering each round ----------------------
    dmid = 0.5 * (grids.d[:-1] + grids.d[1:])
    for n in range(1, spec.rounds + 2):
        if n == 1:
            pl = sol.entry1
            lbl, lw, color = "entering round 1", 2.0, "tab:blue"
        elif n <= spec.rounds:
            pl = sol.Wtables[n][ia0]
            lbl, lw = f"entering round {n}", 1.6
            color = plt.cm.viridis(0.25 + 0.5 * (n - 1) / spec.rounds)
        else:
            pl = sol.Wtables[n][ia0]
            lbl, lw, color = "settlement (undiscounted kink at T)", 2.0, "tab:red"
        slope = np.diff(pl[ci]) / grids.dd
        ax1.plot(dmid / 1000, slope, lw=lw, label=lbl, color=color)
    ax1.axvline(spec.T / 1000, ls="--", color="grey", lw=1)
    ax1.annotate("target T", (spec.T / 1000 + 2, ax1.get_ylim()[0]), fontsize=9)
    ax1.set_xlabel(f"{t_sym} banked, d ({t_sym}k)")
    ax1.set_ylabel(f"marginal value of one more {t_sym} ({i_sym} per {t_sym})")
    # the slope itself is side-specific: the seller divides by the settlement
    # rate, the buyer multiplies by it
    if spec.side == "A":
        ax1.set_title("Marginal value of a banked $:\n"
                      "worth (1+B)/rate when short of T, 1/rate when past it")
    else:
        ax1.set_title("Marginal value of a banked GBP:\n"
                      "worth (1+B) x rate when short of T, the rate past it")
    ax1.legend(fontsize=8)

    # ---- right: one-shot execution price by round (deadline aggression) ------
    rounds = list(range(1, spec.rounds + 1))
    zstars = []
    for n in rounds:
        a_use = spec.params.a0
        act, _ = greedy_offer_action(sol, n, 1, NO_FLOOR, grids.n_offer,
                                     spec.L, 0.0, a_use)
        zstars.append(grids.z[act[0]] if act is not None else np.nan)
    # z on the grid points TOWARD GREED for both sides (the seller's grid
    # reads upward from the anchor, the buyer's downward), so this panel is
    # comparable across sides -- but the one-shot reference is NOT the same
    # number: it is v0's own optimum for THIS side, computed rather than
    # hardcoded (+1.44 sd for the seller, 1.75 sd for the buyer).
    from v0_game import closed_form_optimal_rate
    p0 = spec.params
    zref = abs(closed_form_optimal_rate(p0.a0, p0.sd, p0.bdc_fee, spec.side)
               - p0.a0) / p0.sd
    ax2.step(rounds, zstars, where="mid", marker="o", lw=2, color="tab:orange",
             label="last-offer z* (sd from anchor, greedy direction)")
    ax2.axhline(zref, ls=":", color="grey",
                label=f"single-round optimum |z*| = {zref:.2f} (v0)")
    ax2.set_xlabel("round")
    ax2.set_ylabel("last-offer distance from anchor (sd; higher = holds out)")
    ax2.set_title("Last-offer risk decreases over time (round 4 -> the one-shot v0 price)")
    ax2.set_xticks(rounds)
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_value_surface(sol, save_path="v2_fig2_value.png"):
    """The round-1 value matrix: what any (c, d) start would be worth."""
    spec, grids = sol.spec, sol.grids
    i_sym, t_sym = ("GBP", "$") if spec.side == "A" else ("$", "GBP")
    fig, ax = plt.subplots(figsize=(7.2, 5))
    pl_pct = (sol.entry1 - spec.L) / spec.L * 100.0
    cs = ax.contourf(grids.d / 1000, grids.c / 1000, pl_pct, levels=25, cmap="RdYlGn")
    ax.contour(grids.d / 1000, grids.c / 1000, pl_pct, levels=[0.0], colors="k",
               linewidths=1)
    ax.axvline(spec.T / 1000, color="k", ls="--", lw=1)
    ax.plot(0, spec.L / 1000, marker="*", color="k", ms=14)
    ax.annotate("the actual start (L, 0)", (2, spec.L / 1000 * 0.96),
                fontsize=9)
    ax.set_xlabel(f"{t_sym} banked, d ({t_sym}k)")
    ax.set_ylabel(f"{i_sym} held, c ({i_sym}k)")
    ax.set_title("Round-1 value of every starting book (P/L %, black = flat)")
    fig.colorbar(cs, ax=ax, label="expected P/L (%)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mc_vs_clairvoyant(sol, res, save_path="v2_fig3_regret.png"):
    spec = sol.spec
    W_cv = clairvoyant_wealth(spec, res["X"], res["a5"])
    pl_dp = (res["W"] - spec.L) / spec.L * 100
    pl_cv = (W_cv - spec.L) / spec.L * 100
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bins = np.linspace(min(pl_dp.min(), 0) - 0.5, pl_cv.max() + 0.5, 80)
    ax.hist(pl_dp, bins=bins, alpha=0.65, label=f"DP policy "
            f"(mean {pl_dp.mean():+.2f}%)", color="tab:blue")
    ax.hist(pl_cv, bins=bins, alpha=0.55, label=f"clairvoyant "
            f"(mean {pl_cv.mean():+.2f}%)", color="tab:orange")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("final P/L (%)")
    ax.set_ylabel("paths")
    regret = (W_cv - res["W"]).mean()
    i_sym = "GBP" if spec.side == "A" else "$"
    ax.set_title(f"Optimal play vs hindsight: regret "
                 f"{i_sym} {regret:,.0f} per game "
                 f"({regret / spec.L * 100:.2f}% of capital)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return regret


# ==============================================================================
# 11. RUN
# ==============================================================================
 
if __name__ == "__main__":
    print("=" * 74)
    print("V2: the full game -- exact DP, validation gates, and the benchmark")
    print("=" * 74)
 
    print("\n[0] Terminal layer vs the rules' worked examples")
    terminal_examples()
 
    print("\n[1] Gate 1: one round + all-or-nothing + dollar objective == v1")
    gate1_v1_anchor(K=3)
    print("  gate 1 PASSED: v2 collapses to v1 (and v0 through it)")
 
    print("\n[3] Headline solve")
    spec = TraderSpec()
    i_sym, t_sym = ("GBP", "$") if spec.side == "A" else ("$", "GBP")
    print(f"  card: L={i_sym} {spec.L:,.0f}, T={t_sym} {spec.T:,.0f}, "
          f"A={spec.A:.0%}, B={spec.B:.0%}, {spec.rounds} rounds, K={spec.K}")
    t0 = time.time()
    sol = solve_v2(spec, print_progress=True)
    print(f"  solved in {time.time() - t0:.0f}s")
    print(f"  E[final wealth] = {i_sym} {sol.value:,.1f}   "
          f"expected P/L = {sol.pl() * 100:+.3f}%")
 
    print("\n[4] Gates 3 and 4: Monte Carlo replay through the shared rules")
    res = gate3_and_4(sol, n_paths=20_000, seed=1)
    print("  gates 3 and 4 PASSED")
 
    print("\n[5] Gate 5: the DP is bracketed by the floor and the ceiling")
    gate5_brackets(sol, res)
    print("  gate 5 PASSED")
 
    print("\n[6] Regret figure (optimal play vs the clairvoyant)")
    plot_mc_vs_clairvoyant(sol, res)
    plot_kink_and_policy(sol)
    plot_value_surface(sol)
    print("  wrote v2_fig3_regret.png, v2_fig1_policy.png, v2_fig2_value.png")
 
    # ---- THE MAIN OUTPUT: the optimal strategy, drawn as the path it follows -
    print("\n" + "=" * 74)
    print("[7] THE OPTIMAL STRATEGY for this trader (the path the optimum "
          "follows)")
    print("=" * 74)
    # reuse the already-solved `sol` and replay draws `res` (avoids re-solving).
    _trace = _pick_representative_game(sol, res)
    print()
    print_game_trace(_trace)
    plot_game(_trace, save_path="optimal_strategy.png",
              title=f"Optimal strategy  |  P/L = {_trace['pl']*100:+.2f}%  "
                    f"(L={spec.L:,.0f}, T={spec.T:,.0f})")