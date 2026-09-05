import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))           # Mechanics/
"""
  mm_diagnostics.py -- the figures for the learned market maker, drawn ONLY
  from mm_eval.npz (written by mm_train.py). Nothing is recomputed here and
  nothing is plotted anywhere else. Three figures per side:

    mm_level.png            THE phase-1 figure: what the MM LEARNED, as
                            opposed to what it earned. Top: the threshold
                            it OPERATES, per round -- the P/X cut fitted to
                            its verdicts -- with the emitted level's 10-90%
                            band behind it in grey (the verdicts do not
                            constrain the level away from the margin, so
                            the two differ), the naive break-even (1), the
                            fee-aware one when there is a fee, and the
                            CARRY break-even. Bottom: the Bureau repay
                            fraction on books that actually hold a short.
                            The flatten-on-A / carry-on-B split is visible
                            nowhere else.
    mm_pnl_bars.png         learner and rules on the 200k eval paths with
                            the paired learner-minus-rule gap. MM only.
    mm_learning_curve.png   hard-game P/L against iteration with the
                            hardcoded rules as horizontal lines. A METHODS
                            figure: it shows the rule being FOUND from a
                            deliberately wrong start, not handed over by
                            the initialisation. Convergence itself is read
                            off mm_pnl_bars, which is paired; the lines
                            here are unpaired and the carry ones are soft.

  THE CARRY BREAK-EVEN, corrected. The line previously drawn here was
  1/(1 + (5-n) sd^2/X^2) -- the Jensen term alone, which is the carry
  break-even at fee ZERO -- and it was drawn on side B only. A carrier
  meets the fee at the round-5 forced clear, so its break-even carries the
  fee too, and the two effects are one product:

      carry break-even  =  fee-aware cut  /  (1 + (5-n) sd^2/X^2)

  Checked against the game's own arithmetic (mm_bureau replayed on a single
  carried fill): side A round 1, formula 0.9837 vs measured 0.9839; side B
  round 1, 1.0037 vs 1.0033; agreement to ~4e-4 across all four rounds. At
  fee 0 it collapses to the old expression, so nothing about the fee-0 runs
  changes. It is now drawn on BOTH sides -- on A the fee and the settlement
  term push the cut the same way (0.9837, well below both rules), on B they
  oppose and largely cancel (1.0037, just above naive and far below
  fee-aware). That single line is why naive+carry beats fee-aware+carry on
  side B and loses to it on side A.

  The trader's P/L against each MM is reported in mm_train's console line
  and kept in mm_eval.npz; it is not plotted -- four bars sitting inside
  each other's error bars said less than the sentence does.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Mechanics.fx_mechanics import results_path
import mm_train as M
from mm_train import operating_threshold, repay_on_short_books

# ============================ SETTINGS ======================================
SIDES = ("A", "B")
MM_FEE = M.MM_FEE       # which runs to read: mm_phase1/mm_{side}_fee{MM_FEE}.
                        # FOLLOWS mm_train's setting, so the two cannot drift
                        # apart -- a second copy here silently re-plotted an
                        # OLD fee-0 run after the trainer moved to 1%, which
                        # is worse than an error because it looks like it
                        # worked. Put a number here only to re-plot an
                        # earlier fee on purpose.
# ============================================================================

BP = 1e4


def load(side):
    run = results_path(f"mm_phase1/mm_{side}_fee{MM_FEE:g}_TFA")
    assert (run / "mm_eval.npz").exists(), (
        f"no run to plot for side {side} at MM fee {MM_FEE:g} ({run} has no "
        f"mm_eval.npz): run mm_train.py with MM_FEE = {MM_FEE:g} first, or "
        f"set MM_FEE in this file to a fee you have already trained")
    data = np.load(run / "mm_eval.npz", allow_pickle=False)
    loaded = {key: data[key] for key in data.files}
    # the run's own record of the fee it was trained at, not the folder name
    assert float(loaded["fee"]) == MM_FEE, (
        f"{run}/mm_eval.npz was written at fee {float(loaded['fee'])}, not "
        f"{MM_FEE} -- the folder and its contents disagree")
    return run, loaded


def rounds_with_volume(d, share=0.01):
    #rounds where the trader offers at least `share` of the total volume: the
    #level is identified only there (side A offers nothing after round 1)
    volume = d["record_offered_amount"].sum(axis=0)
    return [r for r in range(volume.size) if volume[r] > share * volume.sum()]


def fee_aware_cut(side, fee):
    #the cut a FLATTENER needs: the fee and nothing else
    return (1.0 - fee) if side == "A" else 1.0 / (1.0 - fee)


def carry_cut(side, fee, sd, X, n_rounds):
    #the cut a CARRIER needs, per round: the flattener's cut discounted by
    #the Jensen term it picks up settling at a5 instead of X. One product,
    #both effects. See the module docstring for the arithmetic check.
    return [fee_aware_cut(side, fee)
            / (1.0 + (n_rounds + 1 - (r + 1)) * sd ** 2 / (X[:, r] ** 2).mean())
            for r in range(n_rounds)]


def plot_level(run, d):
    side, fee, sd = str(d["side"]), float(d["fee"]), float(d["sd"])
    volume = d["record_offered_amount"]
    price_over_rate = d["record_quoted_price"] / d["record_hidden_rate"]
    fill = d["record_fill_weight"]
    level = d["record_break_even"] / d["record_hidden_rate"]
    X = d["record_hidden_rate"]
    n_rounds = volume.shape[1]
    live = rounds_with_volume(d)
    repay = repay_on_short_books(
        {key.replace("record_", ""): d[key] for key in d
         if key.startswith("record_")})

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True)
    for r in range(n_rounds):
        if volume[:, r].sum() <= 0:
            continue
        colour = "k" if r in live else "0.6"
        # the EMITTED level's volume-weighted 10-90% band, for reference:
        # away from the margin the verdicts do not constrain it
        w = volume[:, r] / volume[:, r].sum()
        sorted_level = np.sort(level[:, r])
        cum = np.cumsum(w[np.argsort(level[:, r])])
        lo = sorted_level[min(np.searchsorted(cum, 0.1), sorted_level.size - 1)]
        hi = sorted_level[min(np.searchsorted(cum, 0.9), sorted_level.size - 1)]
        top.plot([r + 1, r + 1], [lo, hi], color="0.8", lw=5, solid_capstyle="butt",
                 zorder=1)
        # the threshold the MM OPERATES: fitted to the verdicts
        threshold, stray = operating_threshold(price_over_rate[:, r],
                                               volume[:, r], fill[:, r], side)
        top.plot(r + 1, threshold, "o", color=colour, zorder=3)
        top.annotate(f"{threshold:.4f}", (r + 1, threshold),
                     textcoords="offset points", xytext=(8, 4), fontsize=8,
                     color=colour)
    top.plot([], [], "o", color="k", label="operating threshold (fitted to the verdicts)")
    top.plot([], [], color="0.8", lw=5, label="emitted level, 10-90% band over offers")
    top.axhline(1.0, color="tab:blue", lw=1.1, label="naive break-even: the true rate")
    if fee > 0:
        cut = fee_aware_cut(side, fee)
        top.axhline(cut, color="tab:red", lw=1.1, ls="-",
                    label=f"fee-aware break-even {cut:.4f} (flatten)")
    # the carry break-even, BOTH sides: fee and settlement at a5 together
    carry = carry_cut(side, fee, sd, X, n_rounds)
    top.plot(range(1, n_rounds + 1), carry, "s--", color="tab:green", ms=4,
             lw=1.0, label="carry break-even: fee-aware / (1+(5-n) sd^2/X^2)")
    top.set_ylabel("threshold / true rate")
    top.set_title(f"side {side}, MM fee {fee}: the verdict threshold by round "
                  f"(grey dot: no real offer volume)", fontsize=10)
    top.legend(fontsize=8)
    top.grid(alpha=0.3)

    # the repay panel is the flatten/carry finding; side B's fractions are
    # ~0.01, invisible against a fixed 0-1 axis, so the scale follows the
    # data and every bar is labelled with its value
    finite = repay[np.isfinite(repay)]
    peak = float(finite.max()) if finite.size else 1.0
    bottom.bar(range(1, n_rounds + 1), repay, color="0.4", width=0.5)
    bottom.axhline(1.0 if side == "A" else 0.0, color="tab:green", lw=1.1, ls="--",
                   label="predicted: flatten (A) / carry (B)")
    bottom.set_ylim(0, 1.05 if peak > 0.1 else max(peak * 1.6, 0.02))
    for r in range(n_rounds):
        if np.isfinite(repay[r]):
            bottom.annotate(f"{repay[r]:.3f}", (r + 1, repay[r]),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8)
    bottom.set_xlabel("round")
    bottom.set_ylabel("repay fraction on short books")
    bottom.legend(fontsize=8)
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "mm_level.png", dpi=150)
    plt.close(fig)


def plot_pnl_bars(run, d):
    names = ["learned MM"] + [str(n) for n in d["rule_names"]]
    mm = [d["eval_mm_pl"]] + list(d["eval_rule_mm_pl"])
    n = mm[0].size

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    means = [x.mean() * BP for x in mm]
    errors = [2 * x.std(ddof=1) / np.sqrt(n) * BP for x in mm]
    colours = ["k"] + ["tab:red" if "fee" in nm else "tab:blue" for nm in names[1:]]
    ax.bar(names, means, yerr=errors, color=colours, capsize=4, alpha=0.85)
    for i, rule_pl in enumerate(mm[1:], start=1):
        gap = mm[0] - rule_pl
        ax.annotate(f"learner - rule\n{gap.mean() * BP:+.2f} +/- "
                    f"{2 * gap.std(ddof=1) / np.sqrt(n) * BP:.2f} bp",
                    (i, means[i] + errors[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set_ylabel("MM P/L, bp of the game's capital (+/- 2 se)")
    ax.set_title(f"side {d['side']}, fee {float(d['fee'])}: MM, {n:,} paired paths",
                 fontsize=10)
    ax.set_ylim(top=max(m + e for m, e in zip(means, errors)) * 1.30)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(run / "mm_pnl_bars.png", dpi=150)
    plt.close(fig)


def plot_learning_curve(run, d):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(d["curve_iterations"], d["curve_hard_pl"] * BP, color="k", lw=1.6,
            label="learned MM, hard game (validation paths)")
    styles = {1.0: "-", 0.0: "--"}
    # each line is a VAL_PATHS estimate, so it comes with a band. The carry
    # rules ride the whole book to round 5 and are ~5x softer than the
    # flatten ones, which is why a curve can sit "on" a dashed line and
    # still wander: read convergence off the PAIRED eval (mm_pnl_bars),
    # never off the gap between this curve and these lines.
    errors = (d["val_rule_se"] if "val_rule_se" in d
              else np.zeros_like(d["val_rule_pl"]))
    for name, repay, value, err in zip(d["rule_names"], d["rule_repay"],
                                       d["val_rule_pl"], errors):
        colour = "tab:red" if "fee" in str(name) else "tab:blue"
        ax.axhline(value * BP, ls=styles.get(float(repay), ":"), lw=1.1,
                   color=colour,
                   label=f"{name} {value * BP:+.2f}"
                         + (f" +/- {err * BP:.2f} bp" if err > 0 else " bp"))
        if err > 0:
            ax.axhspan((value - err) * BP, (value + err) * BP, color=colour,
                       alpha=0.10, lw=0)
    ax.axvline(int(d["best_iteration"]), color="0.6", lw=0.8, ls=":",
               label=f"best checkpoint (iter {int(d['best_iteration'])})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MM P/L, bp of the game's capital")
    ax.set_title(f"side {d['side']}, MM fee {float(d['fee'])}: learning curve "
                 f"against the hardcoded rules")
    # the rule-recovery trace on the right axis: the share of offered volume
    # on which the hard verdicts match the benchmark rule
    twin = ax.twinx()
    # which rule the verdicts were scored against, as recorded by the run
    benchmark = (str(d["benchmark_name"]) if "benchmark_name" in d
                 else ("naive" if float(d["fee"]) == 0.0 else "fee-aware"))
    twin.plot(d["curve_iterations"], d["curve_agree"] * 100, color="0.6",
              lw=0.9, label=f"agreement with the {benchmark} rule (right)")
    twin.set_ylabel(f"verdicts agreeing with the {benchmark} rule, % of volume",
                    color="0.4", fontsize=8)
    twin.set_ylim(0, 102)
    twin.tick_params(axis="y", labelsize=7, colors="0.4")
    lines = [line for line in ax.get_lines() + twin.get_lines()
             if not line.get_label().startswith("_")]
    # below the axes: in the panel it covered the rule lines it exists to
    # label, and on side A the two carry lines sat underneath it
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "mm_learning_curve.png", dpi=150)
    plt.close(fig)


def main():
    for side in SIDES:
        run, d = load(side)
        plot_level(run, d)
        plot_pnl_bars(run, d)
        plot_learning_curve(run, d)
        print(f"side {side}, MM fee {MM_FEE:g}: three figures in {run}/  "
              f"(trader {d['trader_fingerprint']})")


if __name__ == "__main__":
    main()