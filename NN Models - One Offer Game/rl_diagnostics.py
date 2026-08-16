import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
================================================================================
 rl_diagnostics.py -- shared ground truth and the five outputs
================================================================================
 Imported by PPO_train.py (SB3) and torch_train.py (hand-rolled). Every
 function takes SPEC as an argument: the torch arm climbs a rounds ladder, so
 the game cannot be a module-level constant here. Only main() -- the PPO
 report at the bottom -- defaults to the env's own spec.

 FIVE OUTPUTS, nothing else:
   representative_path.png   the median-P/L game, learner beside the DP on the
                             SAME path, drawn by v2's own plot_game
   set_path.png              the same figure on the FIXED path in SET_PATH --
                             identical in every run, so policies can be
                             compared across runs, arms and days. The actions
                             taken on it are also printed in the summary.
   learning_curve.png        P/L against episode, with the DP and always-BdC
                             baselines drawn as horizontal lines
   pl_hist.png               the P/L distribution, both baselines marked, with
                             an inset zoom (the spread is ~100x the gap between
                             the policies, so the main panel cannot show it)
   summary.txt               mean P/L and both baselines, the first offer, the
                             per-round offer table

 GROUND TRUTH (what the learners are scored against):
   solve_anchors   solve the v2 DP fresh, return it with the always-BdC floor,
                   the DP's fresh-book quote per round, and the REPLAYED value
                   plus the paired edge. Never the table value: backward
                   induction maximises over a grid at every layer and the
                   error compounds upward.
   dp_actions      replay the DP along given paths -> raw env actions, episode
                   records, P/L
   dump_all_pl     the always-BdC counterfactual, closed form, per path.
                   LOAD-BEARING: the torch arm's EXCESS reward subtracts it,
                   so this file is a dependency of training, not just of
                   reporting. gate_dump_all checks it against the env.

 No stable-baselines3 anywhere in this file: it measures policies, it does not
 load or train them. PPO's re-report entry point lives in PPO_train.py, so the
 dependency runs one way only (trainers -> diagnostics).

 K = 1 only: with one offer per round the belief bracket is always fresh at
 the offer step, so the DP's greedy action needs no bracket-index translation
 (the K > 1 mapping lives in rl_env.gate_G3). Asserted, not assumed.
================================================================================
"""

# ============================== SETTINGS ======================================
# Defaults for report(); the library functions take their arguments from the
# caller and read nothing here.
N_PATHS     = 20_000      # paired Monte Carlo paths
SEED        = 123         # path draws only (every policy shares the paths)
GATE_EPISODES = 500       # episodes recorded from the replay report() already
                          # runs, so gate_terminal_wealth costs nothing extra
SET_PATH    = [1.30, 1.36, 1.37, 1.41, 1.33]
                          # the comparison path: the first `rounds` entries
                          # are the reveals X1..XR, the LAST entry is the
                          # settlement a_end (a0 comes from the spec). With
                          # fewer rounds, the middle is dropped and the same
                          # settlement kept, so every rung of the ladder is
                          # judged on the same scenario.
# ==============================================================================

from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rl_env import Game, P_MIN, P_MAX
from Mechanics.fx_mechanics import bdc_payoff_per_unit, trade_rate, results_path
from DP_Models.v0_game import closed_form_optimal_rate
from DP_Models.v2_multiple_rounds import (NO_FLOOR, _rate_at, bdc_baseline,
                                greedy_bdc_action, greedy_offer_action,
                                plot_game, solve_v2, terminal_wealth)


PATH_PNG  = "representative_path.png"
SET_PNG   = "set_path.png"
CURVE_PNG = "learning_curve.png"
HIST_PNG  = "pl_hist.png"
SUMMARY   = "summary.txt"


def _greed(spec):
    """+1 sells the initially held currency. The buyer's quote mirrors, the
    same reflection _rate_at applies (A: P = a + sd*z, B: P = a - sd*z)."""
    return 1.0 if spec.side == "A" else -1.0


def _check_K1(spec):
    assert spec.K == 1, (
        f"rl_diagnostics supports one offer per round; this spec has K = "
        f"{spec.K}. The K > 1 bracket translation lives in rl_env.gate_G3.")


# ==============================================================================
# 1. COMMON RANDOM NUMBERS
# ==============================================================================
# The rate path does not depend on the policy (X_n = X_{n-1} + sd*eps_n however
# the trader acts), so pre-drawing it and forcing it through reset(options=)
# pairs every policy on identical randomness. That is the only reason the
# ~1e-4 prize is measurable at all: unpaired, the error bar is ~1e-3.

def draw_paths(spec, n_paths, seed):
    """X[:, r] is round r+1's hidden rate; a_end the settlement draw."""
    p = spec.params
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_paths, spec.rounds + 1))
    levels = p.a0 + np.cumsum(p.sd * eps, axis=1)
    return levels[:, :spec.rounds], levels[:, spec.rounds]


# ==============================================================================
# 2. GROUND TRUTH
# ==============================================================================

def dump_all_pl(spec, X1, a_end):
    """Always-BdC, per path, closed form: the whole book goes at the round-1
    BdC rate and the proceeds settle. gate_dump_all checks it against the env
    playing the same policy, to 1e-9."""
    X1, a_end = np.asarray(X1, float), np.asarray(a_end, float)
    banked = spec.L * bdc_payoff_per_unit(X1, spec.params.bdc_fee, spec.side)
    r_end = 1.0 / a_end if spec.side == "A" else a_end
    W = (banked - spec.B * np.maximum(spec.T - banked, 0.0)) * r_end
    return (W - spec.L) / spec.L


def make_dp_policy(spec, sol):
    """The v2 optimum as an env-side policy: greedy from the solved tables."""
    _check_K1(spec)
    grids, greed = sol.grids, _greed(spec)

    def pol(obs, env):
        held, banked = (env.c, env.d) if spec.side == "A" else (env.d, env.c)
        if env.phase == 0:
            assert env.lo == -np.inf and env.hi == np.inf, \
                "bracket not fresh -- K must be 1 here"
            act, _ = greedy_offer_action(sol, env.n, 1, NO_FLOOR,
                                         grids.n_offer, held, banked, env.a)
            if act is None:                       # the DP stops: offer nothing
                return np.array([1.0, 0.0], dtype=np.float32)
            pi, q = act
            P = _rate_at(spec, env.a, grids.z[pi])      # $/GBP, the slip quote
            return np.array([P, (q / held if held > 0 else 0.0) * greed],
                            dtype=np.float32)
        m = greedy_bdc_action(sol, env.n, held, banked, env.X)
        return np.array([1.0, (m / held if held > 0 else 0.0) * greed],
                        dtype=np.float32)
    return pol


def bdc_policy_for(spec):
    """Offer nothing; dump the whole book at the first BdC. After round 1 the
    dump is a no-op (nothing left), so one line covers every step."""
    greed = _greed(spec)

    def pol(obs, env):
        return np.array([1.0, 0.0 if env.phase == 0 else greed],
                        dtype=np.float32)
    return pol


def fresh_book_offers(spec, sol):
    """The DP's quote entering round n holding the FULL book at the opening
    anchor. Round 1's entry is the exact optimal first offer of the game; the
    rest are the counterfactual 'if you still held everything'. None where the
    DP declines to offer."""
    _check_K1(spec)
    return [None if (act := greedy_offer_action(sol, n, 1, NO_FLOOR,
                                                sol.grids.n_offer, spec.L, 0.0,
                                                spec.params.a0)[0]) is None
            else float(_rate_at(spec, spec.params.a0, sol.grids.z[act[0]]))
            for n in range(1, spec.rounds + 1)]


def dp_actions(spec, sol, X, a_end):
    """Replay the DP along the given paths. Returns (acts, eps, pl): raw
    env-style actions so a second engine can be driven through identical
    decisions, the episode records, and the replayed P/L."""
    return run_paths(spec, make_dp_policy(spec, sol), X, a_end,
                     label="DP", record=True)[:3]


def solve_anchors(spec, n_paths=20_000, seed=4242, print_progress=True):
    """Solve the DP fresh and measure everything the learners are scored
    against. Returns (sol, bdc_pl, dp_offers, rep).

    rep['replay'] is the DP's replayed value, rep['edge'] the paired
    DP-minus-always-BdC difference on shared paths -- the prize a learner
    actually has to find."""
    _check_K1(spec)
    sol = solve_v2(spec, print_progress=print_progress)
    _, bdc_pl = bdc_baseline(spec)                      # analytic floor
    dp_offers = fresh_book_offers(spec, sol)
    X, a_end = draw_paths(spec, n_paths, seed)
    _, _, pl = dp_actions(spec, sol, X, a_end)
    edge = pl - dump_all_pl(spec, X[:, 0], a_end)       # paired, same paths
    # THE REPORTED DP VALUE IS FLOOR-ANCHORED (a control variate):
    #     value = analytic always-BdC floor + paired (DP - dump) edge.
    # The raw MC mean of pl carries ~ +/-0.11% at 20k paths -- wide enough to
    # print BELOW the floor by chance (observed), and to disagree with every
    # other raw mean of the same quantity. The paired edge carries ~ +/-0.007%
    # because the two policies differ only on filled paths, so anchoring the
    # level to the exactly-known floor inherits that tight bar.
    rep = dict(table=float(sol.pl()),
               replay=float(bdc_pl + edge.mean()),
               replay_se=float(edge.std(ddof=1) / np.sqrt(n_paths)),
               replay_raw=float(pl.mean()),
               edge=float(edge.mean()),
               edge_se=float(edge.std(ddof=1) / np.sqrt(n_paths)),
               n_paths=n_paths, seed=seed)
    if print_progress:
        print(f"  DP value {rep['replay'] * 100:+.4f}% +/- "
              f"{2 * rep['replay_se'] * 100:.4f}  (floor-anchored: analytic "
              f"floor {bdc_pl * 100:+.4f}% + paired edge "
              f"{rep['edge'] * 100:+.4f}%)")
        print(f"  DP table {rep['table'] * 100:+.4f}% -- inflated (anchor-grid "
              f"bias); raw 20k replay mean {rep['replay_raw'] * 100:+.4f}% "
              f"carries +/-{2 * (pl.std(ddof=1)/np.sqrt(n_paths)) * 100:.4f} "
              f"and is not the quotable number")
    return sol, bdc_pl, dp_offers, rep


# ==============================================================================
# 3. REPLAY THROUGH THE CERTIFIED ENV
# ==============================================================================

def run_paths(spec, policy, X, a_end, label="policy", record=False,
              n_record=None, progress_every=5000):
    """Drive `policy` through rl_env.Game on the given paths.

    Returns (acts, eps, pl, stats). `acts` is always recorded; `eps` holds full
    episode records for the first n_record paths when record=True; `stats`
    carries the per-round arrays the summary table uses."""
    _check_K1(spec)
    R, greed = spec.rounds, _greed(spec)
    n_paths = X.shape[0]
    n_record = n_paths if n_record is None else min(n_record, n_paths)
    env = Game(spec)

    acts = np.zeros((n_paths, 2 * R, 2))
    pl = np.empty(n_paths)
    z = np.full((n_paths, R), np.nan)      # quote distance (sd, greed +)
    frac = np.zeros((n_paths, R))          # signed offered fraction
    fill = np.zeros((n_paths, R), bool)    # offer shown AND filled
    shown = np.zeros((n_paths, R), bool)   # a real, fillable offer was shown
    dump = np.zeros((n_paths, R))          # signed BdC fraction (- = buy back)
    eps = []

    for i in range(n_paths):
        obs, _ = env.reset(options={"path": X[i].tolist(),
                                    "a_end": float(a_end[i])})
        keep = record and i < n_record
        rec = dict(X=[float(v) for v in X[i]], a_end=float(a_end[i]),
                   offers=[], sizes=[], q_off=[], fills=[], dumps=[],
                   dump_amt=[], held=[], banked=[]) if keep else None
        if keep:
            h, b = (env.c, env.d) if spec.side == "A" else (env.d, env.c)
            rec["held"].append(h / spec.L)
            rec["banked"].append(b / spec.T)
        done, total, step = False, 0.0, 0
        while not done:
            act = np.asarray(policy(obs, env), dtype=np.float32)
            acts[i, step] = act
            n = env.n
            P = float(np.clip(act[0], P_MIN, P_MAX))
            sf = float(np.clip(act[1], -1.0, 1.0))
            held_now = env.c if sf >= 0 else env.d
            if env.phase == 0:
                if held_now > 1e-6 and abs(sf) > 1e-6:
                    shown[i, n - 1] = True
                    z[i, n - 1] = (P - env.a) / spec.params.sd * greed
                    frac[i, n - 1] = sf
                c0 = env.c
                obs, r, done, _, _ = env.step(act)
                filled = shown[i, n - 1] and abs(env.c - c0) > 1e-9
                fill[i, n - 1] = filled
                if keep:
                    rec["offers"].append(P)
                    rec["sizes"].append(sf)
                    rec["q_off"].append(abs(sf) * held_now)
                    rec["fills"].append(bool(filled))
            else:
                dump[i, n - 1] = sf if held_now > 1e-6 else 0.0
                obs, r, done, _, _ = env.step(act)
                if keep:
                    rec["dumps"].append(sf)
                    rec["dump_amt"].append(abs(sf) * held_now)
            if keep:
                h, b = (env.c, env.d) if spec.side == "A" else (env.d, env.c)
                rec["held"].append(h / spec.L)
                rec["banked"].append(b / spec.T)
            total += r
            step += 1
        pl[i] = total
        if keep:
            rec["pl"] = float(total)
            eps.append(rec)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {label}: {i + 1:,}/{n_paths:,} paths")

    return acts, eps, pl, dict(z=z, frac=frac, fill=fill, shown=shown,
                               dump=dump)


def first_offer(spec, policy):
    """The opening offer, read once. At round 1 the observation is fixed --
    full book, nothing banked, anchor a0 -- so a DETERMINISTIC policy makes
    the same first offer in every game, and one query settles what it is.

    Returns None when the policy shows nothing fillable (which is itself the
    finding: a size pointing at the empty pocket is a silent no-op)."""
    env = Game(spec)
    obs, _ = env.reset(options={"path": [spec.params.a0] * spec.rounds,
                                "a_end": spec.params.a0})
    act = np.asarray(policy(obs, env), dtype=np.float64)
    P = float(np.clip(act[0], P_MIN, P_MAX))
    sf = float(np.clip(act[1], -1.0, 1.0))
    held = env.c if sf >= 0 else env.d
    if held <= 1e-6 or abs(sf) <= 1e-6:
        return None
    z = (P - spec.params.a0) / spec.params.sd * _greed(spec)
    from scipy.stats import norm
    return dict(quote=P, z=z, frac=sf, fill_prob=float(norm.sf(z)))


# ==============================================================================
# 4. GATES -- exact, cheap, run before any number is quoted
# ==============================================================================

def gate_dump_all(spec, n_paths=2000, seed=5, verbose=True):
    """dump_all_pl's closed form against the env playing that policy. Exact:
    no statistics, no tolerance beyond floating point."""
    X, a_end = draw_paths(spec, n_paths, seed)
    _, _, pl, _ = run_paths(spec, bdc_policy_for(spec), X, a_end,
                            progress_every=0)
    gap = float(np.max(np.abs(pl - dump_all_pl(spec, X[:, 0], a_end))))
    ok = gap < 1e-9
    if verbose:
        print(f"  gate  dump_all_pl vs the env, {n_paths} paths: max gap "
              f"{gap:.2e}  [{'ok' if ok else 'FAIL'}]")
    return ok


def gate_terminal_wealth(spec, eps, verbose=True):
    """Every episode record's P/L rebuilt through v2's own terminal_wealth."""
    worst = 0.0
    for e in eps:
        # run_paths ALREADY stores role order -- it records initial-currency/L
        # as "held" and target-currency/T as "banked" -- and terminal_wealth
        # takes role order too (rl_env calls it the same way). Reordering here
        # a second time is the identity on side A and swaps the two on side B.
        held, banked = e["held"][-1] * spec.L, e["banked"][-1] * spec.T
        W = terminal_wealth(spec, held, banked, e["a_end"])
        worst = max(worst, abs((W - spec.L) / spec.L - e["pl"]))
    ok = worst < 1e-9
    if verbose:
        print(f"  gate  episode P/L vs v2 terminal_wealth, {len(eps)} games: "
              f"max gap {worst:.2e}  [{'ok' if ok else 'FAIL'}]")
    return ok


# ==============================================================================
# 5. THE THREE FIGURES
# ==============================================================================

def _finish(fig, axes, save_path):
    for ax in np.atleast_1d(axes).ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(save_path, xs, ys, dp_ref, bdc_pl, xlabel="episode",
                        tag="", roll=None):
    """P/L against training progress, with both baselines as horizontal lines.
    The raw series carries exploration noise, so the rolling mean is what to
    read; neither is the trained policy's value, which comes from the paired
    replay in the summary."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(xs, ys, lw=0.7, color="tab:blue", alpha=0.35, label="per episode")
    w = roll or max(len(ys) // 60, 2)
    if len(ys) > w:
        m = np.convolve(ys, np.ones(w) / w, mode="valid")
        ax.plot(xs[w - 1:], m, lw=1.8, color="tab:blue",
                label=f"rolling mean ({w:,})")
    ax.axhline(dp_ref, ls="--", color="tab:green", lw=1.6,
               label=f"DP optimum, replayed ({dp_ref * 100:+.3f}%)")
    ax.axhline(bdc_pl, ls=":", color="grey", lw=1.6,
               label=f"always-BdC ({bdc_pl * 100:+.3f}%)")
    lo = min(np.percentile(ys, 2), bdc_pl, dp_ref)
    hi = max(np.percentile(ys, 98), bdc_pl, dp_ref)
    pad = 0.15 * (hi - lo) + 1e-6
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P/L (fraction of the book)")
    ax.set_title(f"learning curve  {tag}".strip())
    ax.legend(fontsize=8)
    _finish(fig, ax, save_path)


def _eps_to_trace(spec, e):
    """One episode record -> v2's own trace dict, so plot_game can draw it.

    v2's traces are ROLE-ordered: play_game starts (c, d) = (spec.L, 0) and
    hands them to terminal_wealth, and plot_game draws c/L as the initial
    currency draining and d/T as the target banked. run_paths already records
    that order, so "held" IS c and "banked" IS d on BOTH sides -- reordering
    by side here would swap the two curves and cross-divide them on side B."""
    sd, a, greed = spec.params.sd, spec.params.a0, _greed(spec)
    rounds = []
    for n in range(1, spec.rounds + 1):
        j = n - 1
        quote = e["offers"][j]
        c, d = e["held"][2 * n] * spec.L, e["banked"][2 * n] * spec.T
        offers = []
        if abs(e["sizes"][j]) > 1e-6 and e["q_off"][j] > 1e-6:
            h_off, b_off = e["held"][2 * j + 1], e["banked"][2 * j + 1]
            offers.append({"offer_no": 1, "z": (quote - a) / sd * greed,
                           "P": trade_rate(quote, spec.side), "quote": quote,
                           "q": e["q_off"][j], "accepted": bool(e["fills"][j]),
                           "c": h_off * spec.L, "d": b_off * spec.T})
        rounds.append({"n": n, "anchor": a, "X": e["X"][j], "offers": offers,
                       "bdc_dump": e["dump_amt"][j], "c": c, "d": d})
        a = e["X"][j]
    return {"spec": spec, "rounds": rounds, "a5": e["a_end"],
            "c": e["held"][-1] * spec.L, "d": e["banked"][-1] * spec.T,
            "W": spec.L * (1 + e["pl"]), "pl": e["pl"]}


def plot_path(save_path, spec, sol, policy, X, a_end, pl=None, tag="",
              pick="median", label="learner", note=None):
    """One representative game: the learner above, the DP below, on the SAME
    path, drawn by v2's plot_game so the figure matches the thesis format.

    pick='median' takes the path whose learner P/L is the median of `pl` --
    a typical game rather than a flattering or alarming one. Pass pick=int for
    a specific path, or 'random' for a draw."""
    _check_K1(spec)
    if pick == "median" and pl is not None:
        i = int(np.argsort(pl)[len(pl) // 2])
    elif pick == "random":
        i = int(np.random.default_rng(0).integers(len(X)))
    else:
        i = int(pick)
    X1, a1 = X[i:i + 1], a_end[i:i + 1]
    _, l_eps, l_pl, _ = run_paths(spec, policy, X1, a1, record=True,
                                  progress_every=0)
    _, d_eps, d_pl, _ = run_paths(spec, make_dp_policy(spec, sol), X1, a1,
                                  record=True, progress_every=0)
    dump = float(dump_all_pl(spec, X1[:, 0], a1)[0])
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 12))
    for row, (who, e, v) in enumerate(((label, l_eps[0], l_pl[0]),
                                       ("DP optimum", d_eps[0], d_pl[0]))):
        plot_game(_eps_to_trace(spec, e), ax_rate=axes[2 * row],
                  ax_inv=axes[2 * row + 1],
                  title=f"{who}   P/L {v * 100:+.2f}%")
    fig.suptitle(f"{note or f'representative game (median P/L, path #{i})'}"
                 f"   always-BdC on this path {dump * 100:+.2f}%   "
                 f"{tag}".strip(), y=1.001, fontsize=10)
    _finish(fig, axes, save_path)
    return i


def set_path_arrays(spec):
    """SET_PATH -> (X, a_end) arrays for this spec's round count."""
    if SET_PATH is None or len(SET_PATH) < 2:
        return None, None
    xs = SET_PATH[:-1][:spec.rounds]
    if len(xs) < spec.rounds:                    # pad by holding the last rate
        xs = xs + [xs[-1]] * (spec.rounds - len(xs))
    return np.array([xs], float), np.array([SET_PATH[-1]], float)


def trace_on_set_path(spec, policy, label):
    """One line per half-step of what `policy` does on the set path."""
    X, ae = set_path_arrays(spec)
    _, eps, pl, _ = run_paths(spec, policy, X, ae, record=True,
                              progress_every=0)
    e, L = eps[0], []
    a = spec.params.a0
    for k in range(spec.rounds):
        filled = e["fills"][k]
        off = (f"offer {e['offers'][k]:.4f} on {e['q_off'][k]:,.0f} "
               f"({'FILLED' if filled else 'rejected'})"
               if abs(e["sizes"][k]) > 1e-6 and e["q_off"][k] > 1e-6
               else "no offer")
        L.append(f"    r{k+1}  a={a:.2f} X={e['X'][k]:.2f}   {off};   "
                 f"BdC moves {e['dump_amt'][k]:,.0f}")
        a = e["X"][k]
    L.append(f"    settle a_end={e['a_end']:.2f}   P/L {pl[0]*100:+.3f}%")
    return f"  {label} on the set path:\n" + "\n".join(L)


def plot_pl_hist(save_path, pl, dp_ref, bdc_pl, tag="", label="learner"):
    """The P/L distribution with both baselines marked. The inset is not
    decoration: the spread is ~100x the gap between the policies, so the main
    panel physically cannot separate the three lines."""
    pl = np.asarray(pl, float)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(pl * 100, bins=80, color="tab:blue", alpha=0.55, label=label)
    marks = ((pl.mean() * 100, "tab:blue", "-", f"{label} mean"),
             (dp_ref * 100, "tab:green", "--", "DP optimum, replayed"),
             (bdc_pl * 100, "grey", ":", "always-BdC"))
    for v, col, ls, name in marks:
        ax.axvline(v, color=col, ls=ls, lw=1.6, label=f"{name} ({v:+.3f}%)")
    ax.set_xlabel("final P/L (%)")
    ax.set_ylabel("paths")
    ax.set_title(f"P/L over {len(pl):,} paths  {tag}".strip())
    ax.legend(fontsize=8)

    vals = [m[0] for m in marks]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 2.0, 0.02)
    inset = ax.inset_axes([0.60, 0.10, 0.37, 0.30])
    for v, col, ls, _ in marks:
        inset.axvline(v, color=col, ls=ls, lw=1.6)
    inset.set_xlim(lo - pad, hi + pad)
    inset.set_yticks([])
    inset.tick_params(labelsize=7)
    inset.set_xlabel("zoom: the means (%)", fontsize=7)
    _finish(fig, ax, save_path)


# ==============================================================================
# 6. THE SUMMARY
# ==============================================================================

def _pm(x):
    return x.mean(), 2 * x.std(ddof=1) / np.sqrt(len(x))


def write_summary(save_path, spec, pl, dump_pl, rep, bdc_pl, dp_pl=None,
                  stats=None, learner_offer=None, dp_offer=None,
                  dp_offers=None, label="learner", extra=(), seed=None,
                  files=None):
    """The text output. pl and dump_pl must be PAIRED -- the same paths in the
    same order -- or the differences below are meaningless."""
    pl, dump_pl = np.asarray(pl, float), np.asarray(dump_pl, float)
    sym = "GBP" if spec.side == "A" else "$"
    a0, sd = spec.params.a0, spec.params.sd
    L = []
    w = L.append

    def money(name, x):
        m, pm = _pm(x)
        w(f"  {name:<26} {m * 100:+9.4f}% +/- {pm * 100:.4f}%   "
          f"({spec.L * m:+,.0f} +/- {spec.L * pm:,.0f} {sym})")

    w("=" * 78)
    w(f" SUMMARY  |  side {spec.side}, rounds {spec.rounds}, K {spec.K}, "
      f"{len(pl):,} paired paths")
    w("=" * 78)
    w(f" game (from rl_env): L {spec.L:,.0f} {sym}, target {spec.T:,.0f}, "
      f"holding fee {spec.A:.0%}, shortfall {spec.B:.0%}")
    w(f" market: a0 {a0}, sd {sd}, BdC fee {spec.params.bdc_fee:.0%}")
    # PROVENANCE -- so a summary file identifies its own run. Two summaries are
    # only comparable if the path seed and count match; without this line you
    # cannot tell a re-run from a stale copy.
    w(f" provenance: path seed {seed if seed is not None else 'UNRECORDED'}, "
      f"{len(pl):,} paths, written {datetime.now():%Y-%m-%d %H:%M}")

    w("\nMEAN P/L  (levels are floor-anchored: analytic always-BdC floor +")
    w("the paired difference below. Raw 20k MC means carry ~+/-0.11% and can")
    w("even land below the floor by chance; they are not shown.)")
    money(label, bdc_pl + (pl - dump_pl))
    if dp_pl is not None:
        money("DP optimum (replayed)", bdc_pl + (dp_pl - dump_pl))
    else:
        w(f"  {'DP optimum (replayed)':<26} {rep['replay'] * 100:+9.4f}% +/- "
          f"{2 * rep['replay_se'] * 100:.4f}%")
    w(f"  {'always-BdC':<26} {bdc_pl * 100:+9.4f}%  (analytic, exact)")
    w("\n  paired differences (common random numbers -- the only way the"
      " signal clears the noise):")
    money(f"{label} - always-BdC", pl - dump_pl)
    if dp_pl is not None:
        money("DP - always-BdC (prize)", dp_pl - dump_pl)
        money(f"DP - {label} (regret)", dp_pl - pl)
    else:
        w(f"  {'DP - always-BdC (prize)':<26} {rep['edge'] * 100:+9.4f}% +/- "
          f"{2 * rep['edge_se'] * 100:.4f}%")
    w(f"\n  DP table value {rep['table'] * 100:+.4f}% -- inflated by the "
      f"maximisation bias that")
    w(f"  compounds through backward induction. The replayed value above is "
      f"the quotable one.")

    w("\nTHE FIRST OFFER  (round 1's observation is fixed, so a deterministic")
    w("policy makes the SAME opening offer in every single game)")
    for who, o in ((label, learner_offer), ("DP optimum", dp_offer)):
        if o is None:
            w(f"  {who:<12} NO FILLABLE OFFER -- size ~0, or pointing at the "
              f"empty pocket.")
            w(f"  {'':<12} This is a silent no-op, not a strategy: check the "
              f"action mapping.")
        else:
            w(f"  {who:<12} quote {o['quote']:.4f} $/GBP  (z = {o['z']:+.2f} "
              f"sd)  size {abs(o['frac']):.1%} of the book")
            w(f"  {'':<12} fill probability {o['fill_prob']:.1%}")
    zref = abs(closed_form_optimal_rate(a0, sd, spec.params.bdc_fee,
                                        spec.side) - a0) / sd
    w(f"  {'v0 one-shot':<12} |z*| = {zref:.2f} sd  (the closed form, no "
      f"continuation value)")

    if stats is not None:
        w("\nOFFERS BY ROUND  (over the paths; 'shown' = a fillable offer was "
          "made)")
        w(f"  {'round':>5} {'shown%':>7} {'z (sd)':>9} {'size':>7} "
          f"{'fill%':>7} {'BdC (signed)':>13}")
        for n in range(spec.rounds):
            sh = stats["shown"][:, n]
            if sh.any():
                cols = (f"{np.nanmean(stats['z'][sh, n]):+9.3f}",
                        f"{np.abs(stats['frac'][sh, n]).mean():7.3f}",
                        f"{100 * stats['fill'][sh, n].sum() / sh.sum():7.1f}")
            else:
                cols = (f"{'--':>9}", f"{'--':>7}", f"{'--':>7}")
            w(f"  {n + 1:>5} {100 * sh.mean():7.1f} {cols[0]} {cols[1]} "
              f"{cols[2]} {stats['dump'][:, n].mean():+13.3f}")
        if not stats["shown"].any():
            w("  [FLAG] the policy never made a fillable offer on any path. "
              "Its P/L is")
            w("         the always-BdC baseline by construction, and the "
              "action mapping,")
            w("         not the game, is what that measures.")
        if dp_offers is not None:
            w("\n  DP fresh-book quote by round (full book, anchor a0):")
            for n, q in enumerate(dp_offers, start=1):
                w(f"    round {n}: " + ("no offer -- straight to the BdC"
                  if q is None else
                  f"{q:.4f} $/GBP  (z = {(q - a0) / sd * _greed(spec):+.2f} sd)"))
    for line in extra:
        w(line)
    w("\nFILES: " + ", ".join(files if files else (SUMMARY,)))
    text = "\n".join(L) + "\n"
    Path(save_path).write_text(text)
    return text


# ==============================================================================
# 7. THE ONE ENTRY POINT -- both trainers call this
# ==============================================================================

def report(spec, policy, out_dir, label="learner", curve=None, sol=None,
           n_paths=N_PATHS, seed=SEED, tag=None, extra=(), echo=True):
    """The four outputs, for any env-side policy.

    policy   f(obs, env) -> raw env action. Both arms hand over a policy that
             is replayed through the CERTIFIED env, so the two are measured by
             the same engine and their numbers are comparable.
    curve    (xs, ys, xlabel) from training, or None to skip the curve.
    sol      a solved DP to reuse; None solves one fresh (the honest default).

    Returns the summary text."""
    _check_K1(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"side {spec.side}, R{spec.rounds}"

    if sol is None:
        print("ground truth: solving the v2 DP fresh")
        sol, bdc_pl, dp_offers, rep = solve_anchors(spec)
    else:
        _, bdc_pl = bdc_baseline(spec)
        dp_offers = fresh_book_offers(spec, sol)
        X0, a0_ = draw_paths(spec, 20_000, 4242)
        _, _, dppl = dp_actions(spec, sol, X0, a0_)
        edge = dppl - dump_all_pl(spec, X0[:, 0], a0_)
        rep = dict(table=float(sol.pl()), replay=float(dppl.mean()),
                   replay_se=float(dppl.std(ddof=1) / np.sqrt(len(dppl))),
                   edge=float(edge.mean()),
                   edge_se=float(edge.std(ddof=1) / np.sqrt(len(edge))),
                   n_paths=20_000, seed=4242)
    if not gate_dump_all(spec):
        raise SystemExit("a gate FAILED; not reporting numbers on top of it")

    print(f"replaying {label} and the DP on the SAME {n_paths:,} paths")
    X, a_end = draw_paths(spec, n_paths, seed)
    # record the first GATE_EPISODES episodes of the run that happens anyway,
    # so the terminal-wealth gate below costs no extra replay
    _, eps, pl, stats = run_paths(spec, policy, X, a_end, label,
                                  record=True, n_record=GATE_EPISODES)
    if not gate_terminal_wealth(spec, eps):
        raise SystemExit("a gate FAILED; not reporting numbers on top of it")
    _, _, dp_pl, _ = run_paths(spec, make_dp_policy(spec, sol), X, a_end, "DP")
    dump_pl = dump_all_pl(spec, X[:, 0], a_end)
    # one estimate everywhere: the summary's own paired replay supersedes the
    # solve-time one, so the DP value in the tables, the hist line and the
    # learning-curve reference are the SAME number
    e = dp_pl - dump_pl
    rep["replay"] = float(bdc_pl + e.mean())
    rep["replay_se"] = float(e.std(ddof=1) / np.sqrt(len(e)))
    rep["edge"], rep["edge_se"] = float(e.mean()), rep["replay_se"]

    plot_path(out_dir / PATH_PNG, spec, sol, policy, X, a_end, pl, tag,
              label=label)
    written = [PATH_PNG]
    extra = list(extra)
    Xs, aes = set_path_arrays(spec)
    if Xs is not None:
        plot_path(out_dir / SET_PNG, spec, sol, policy, Xs, aes, tag=tag,
                  pick=0, label=label,
                  note="the set path (fixed across runs): a0 "
                       f"{spec.params.a0} -> "
                       + " -> ".join(f"{v:.2f}" for v in SET_PATH))
        written.append(SET_PNG)
        extra += ["\nTHE SET PATH  (fixed across runs; see set_path.png)",
                  trace_on_set_path(spec, policy, label),
                  trace_on_set_path(spec, make_dp_policy(spec, sol), "DP")]
    plot_pl_hist(out_dir / HIST_PNG, pl, rep["replay"], bdc_pl, tag,
                 label=label)
    written.append(HIST_PNG)
    if curve is not None:
        xs, ys, xlabel = curve
        plot_learning_curve(out_dir / CURVE_PNG, xs, ys, rep["replay"],
                            bdc_pl, xlabel, tag)
        written.append(CURVE_PNG)
    written.append(SUMMARY)
    text = write_summary(out_dir / SUMMARY, spec, pl, dump_pl, rep, bdc_pl,
                         dp_pl=dp_pl, stats=stats,
                         learner_offer=first_offer(spec, policy),
                         dp_offer=first_offer(spec, make_dp_policy(spec, sol)),
                         dp_offers=dp_offers, label=label, extra=extra,
                         seed=seed, files=written)
    if echo:
        print("\n" + text)
    print(f"five outputs in {out_dir}/")
    return text