import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))           # Mechanics/
sys.path.insert(0, str(_ROOT / "MM_Phase_1"))   # trader_ and mm_diagnostics,
                                         # whose plotting is reused below
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""
  cotrain_diagnostics.py -- the figures for the co-trained pair, drawn ONLY
  from cotrain_curve.npz and cotrain_eval.npz (written by cotrain.py).
  Nothing is recomputed here and nothing is plotted anywhere else. Three
  figures per side:

    cotrain_curve.png       both hard-game P/Ls against the block index and
                            the four hardcoded rules as MOVING lines (each
                            is P/L against the trader as it stood at that
                            block). The trader panel carries its outside
                            option, always-BdC ON THE SAME PATHS, and its
                            band is the 2 SE of the PAIRED edge over that
                            floor: the raw band carries the rate path and
                            hides everything. Markers say who had just
                            moved. The question the figure answers: do the
                            two settle, or cycle?
    cotrain_policy.png      what the pair LEARNED, block by block: the MM's
                            round-1 level over the true rate against the
                            naive and fee-aware cuts; the trader's round-1
                            quote in $/GBP and the share of its book it
                            puts up; and the share of offered volume the MM
                            fills, learner against the benchmark rule. The
                            last panel is the deadlock detector: a pair
                            that trades nothing learns nothing.
    cotrain_summary.txt     every headline number of the run in one file,
                            as the trader arm's summary.txt: settings,
                            convergence, both P/Ls, the trader's paired
                            edge over always-BdC, the MM against each rule,
                            the policy by round, the set path and the
                            cross-play. No DP row: the exact solution
                            assumes an MM that accepts at the true rate.
    cotrain_crossplay.png   THE equilibrium figure: the 2 x 2 cross-play
                            with Phase 1's pair on the same paths and the
                            four deviation gains, paired. A best response
                            gains nothing, so every gain should be <= 0
                            within its 2 SE. Drawn only when cotrain ran
                            with Phase 1's artefacts present.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Mechanics.fx_mechanics import GameParams, results_path
from NN_one_offer_game.rl_env import GameSpec
import cotrain as C
import MM_Phase_1.mm_diagnostics as mm_diagnostics
import MM_Phase_1.trader_diagnostics as trader_diagnostics

# ============================ SETTINGS ======================================
SIDES = ("A", "B")
MM_FEE = C.MM_FEE       # which runs to read: cotrain/{side}_fee{MM_FEE}.
                        # FOLLOWS cotrain's setting so the two cannot drift
                        # apart (mm_diagnostics' lesson). Put a number here
                        # only to re-plot an earlier fee on purpose
# ============================================================================

BP = 1e4
COLOUR = {"mm": "tab:purple", "trader": "tab:green", "init": "0.5"}


def load(side):
    run = results_path(f"cotrain/{side}_fee{MM_FEE:g}")
    for name in ("cotrain_curve.npz", "cotrain_eval.npz"):
        assert (run / name).exists(), (
            f"no run to plot for side {side} at MM fee {MM_FEE:g} ({run} has "
            f"no {name}): run cotrain.py with MM_FEE = {MM_FEE:g} first")
    curve = np.load(run / "cotrain_curve.npz", allow_pickle=False)
    evaluation = np.load(run / "cotrain_eval.npz", allow_pickle=False)
    curve = {key: curve[key] for key in curve.files}
    evaluation = {key: evaluation[key] for key in evaluation.files}
    assert float(evaluation["fee"]) == MM_FEE, (
        f"{run}/cotrain_eval.npz was written at fee {float(evaluation['fee'])}, "
        f"not {MM_FEE} -- the folder and its contents disagree")
    return run, curve, evaluation


def fee_aware_cut(side, fee):
    #the break-even a fee-aware flattener operates: (1-g) X on A, X/(1-g) on B
    return (1.0 - fee) if side == "A" else 1.0 / (1.0 - fee)


def _who_moved(curve):
    return [str(m) for m in curve["moved"]]


def _mark_movers(ax, x, y, moved):
    #one marker per block, coloured by who had just moved
    for agent, marker in (("mm", "s"), ("trader", "o"), ("init", "D")):
        idx = [i for i, m in enumerate(moved) if m == agent]
        if idx:
            ax.plot(x[idx], y[idx], marker, color=COLOUR[agent], ms=4,
                    ls="none",
                    label=("as initialised" if agent == "init"
                           else f"after the {'MM' if agent == 'mm' else 'trader'} moved"))


def plot_curve(run, curve, evaluation):
    side, fee = str(evaluation["side"]), float(evaluation["fee"])
    x = np.arange(len(curve["mm_pl"]))
    moved = _who_moved(curve)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True)

    # ---- the MM, with the moving rules --------------------------------------
    top.plot(x, curve["mm_pl"] * BP, color="k", lw=1.4,
             label="co-trained MM, hard game (validation paths)")
    top.fill_between(x, (curve["mm_pl"] - curve["mm_2se"]) * BP,
                     (curve["mm_pl"] + curve["mm_2se"]) * BP, color="k",
                     alpha=0.12, lw=0)
    styles = {1.0: "-", 0.0: "--"}
    rule_names = [str(n) for n in curve["rule_names"][0]]
    for j, name in enumerate(rule_names):
        colour = "tab:red" if "fee" in name else "tab:blue"
        repay = float(evaluation["rule_repay"][j])
        top.plot(x, curve["rule_pl"][:, j] * BP, ls=styles.get(repay, ":"),
                 lw=1.0, color=colour,
                 label=f"hardcoded {name} against the trader at that block")
    _mark_movers(top, x, curve["mm_pl"] * BP, moved)
    top.set_ylabel("MM P/L, bp of the game's capital\n(1 bp = 0.01%)")
    top.set_title(f"side {side}, MM fee {fee}: co-training, "
                  f"{int(evaluation['n_inner'])} iterations per block, "
                  f"{str(evaluation['first']).upper()} first")
    top.grid(alpha=0.3)
    top.legend(fontsize=6.5, loc="best")

    # ---- the trader, against its outside option on the SAME paths --------
    paired = "edge_2se" in curve
    if not paired:
        print(f"  {run.name}: cotrain_curve.npz predates the paired floor "
              f"(no edge_2se): raw 2 SE band, no always-BdC line. A fresh "
              f"cotrain.py run adds both")
    band = curve["edge_2se"] if paired else curve["trader_2se"]
    bottom.plot(x, curve["trader_pl"] * 100, color="k", lw=1.4,
                label="co-trained trader, hard game (validation paths)")
    bottom.fill_between(x, (curve["trader_pl"] - band) * 100,
                        (curve["trader_pl"] + band) * 100,
                        color="k", alpha=0.12, lw=0,
                        label=("2 SE of the edge over always-BdC, paired"
                               if paired else "2 SE, raw (old run)"))
    if paired:
        floor = float(curve["bdc_pl"][0])     # constant: fixed paths
        bottom.axhline(floor * 100, color="tab:orange", ls="--", lw=1.4,
                       label=f"always-BdC on the same paths ({floor * 100:+.3f}%)")
    _mark_movers(bottom, x, curve["trader_pl"] * 100, moved)
    bottom.set_ylabel("trader P/L, % of L")
    bottom.set_xlabel("block (0 = the pair as initialised; then alternating)")
    bottom.grid(alpha=0.3)
    bottom.legend(fontsize=6.5, loc="best")
    stopped = int(curve["stopped_at"])
    if stopped >= 0:
        for ax in (top, bottom):
            ax.axvline(x[-1], color="0.6", lw=0.8, ls=":")
        bottom.text(x[-1], bottom.get_ylim()[0], f" flat: stopped at outer "
                    f"step {stopped}", fontsize=7, color="0.4", va="bottom")
    fig.tight_layout()
    fig.savefig(run / "cotrain_curve.png", dpi=150)
    plt.close(fig)


def plot_policy(run, curve, evaluation):
    side, fee = str(evaluation["side"]), float(evaluation["fee"])
    x = np.arange(len(curve["mm_pl"]))
    moved = _who_moved(curve)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
    level, offer, filled = axes

    # ---- the MM's level ---------------------------------------------------
    r1_level = curve["level"][:, 0]
    level.plot(x, r1_level, color="k", lw=1.4,
               label="round-1 level / true rate (median over games)")
    _mark_movers(level, x, r1_level, moved)
    level.axhline(1.0, color="tab:blue", lw=1.0,
                  label="naive break-even: the true rate")
    if fee > 0:
        level.axhline(fee_aware_cut(side, fee), color="tab:red", lw=1.0,
                      label=f"fee-aware break-even {fee_aware_cut(side, fee):.4f}")
    level.set_ylabel("MM level / X, round 1")
    level.grid(alpha=0.3)
    level.legend(fontsize=6.5, loc="best")
    level.set_title(f"side {side}, MM fee {fee}: what the pair learned, "
                    f"block by block")

    # ---- the trader's opening offer ---------------------------------------
    if "offer_rate" in curve:
        r1_offer = curve["offer_rate"]
    else:
        # an older curve saved the quote over a0; a0 is in the eval artefact
        r1_offer = curve["offer"] * float(evaluation["a0"])
        print(f"  {run.name}: cotrain_curve.npz predates offer_rate; the "
              f"quote is rebuilt as offer x a0 = {float(evaluation['a0']):g}")
    offer.plot(x, r1_offer, color="k", lw=1.4,
               label="round-1 quote, $/GBP (median over games)")
    _mark_movers(offer, x, r1_offer, moved)
    twin = offer.twinx()
    twin.plot(x, curve["fraction"] * 100, color="0.5", lw=1.0,
              label="share of the book put up in round 1 (right)")
    twin.set_ylabel("put up, % of L", color="0.4", fontsize=8)
    twin.set_ylim(0, 102)
    twin.tick_params(axis="y", labelsize=7, colors="0.4")
    offer.set_ylabel("trader quote, $/GBP, round 1")
    offer.grid(alpha=0.3)
    lines = [l for l in offer.get_lines() + twin.get_lines()
             if not l.get_label().startswith("_")]
    offer.legend(lines, [l.get_label() for l in lines], fontsize=6.5,
                 loc="best")

    # ---- the flow: the deadlock detector ----------------------------------
    filled.plot(x, curve["filled"] * 100, color="k", lw=1.4,
                label="filled by the co-trained MM")
    filled.plot(x, curve["filled_benchmark"] * 100, color="tab:blue", lw=1.0,
                ls="--", label="filled by the benchmark rule (best rule at that block)")
    filled.plot(x, curve["agree"] * 100, color="0.6", lw=0.9,
                label="verdicts agreeing with the benchmark rule")
    _mark_movers(filled, x, curve["filled"] * 100, moved)
    filled.set_ylabel("% of offered volume")
    filled.set_ylim(-2, 102)
    filled.set_xlabel("block (0 = the pair as initialised; then alternating)")
    filled.grid(alpha=0.3)
    filled.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(run / "cotrain_policy.png", dpi=150)
    plt.close(fig)


def plot_crossplay(run, evaluation):
    names = [str(n) for n in evaluation["eval_names"]]
    if len(names) < 4:
        print(f"  no cross-play in {run}: cotrain ran without Phase 1's "
              f"artefacts; cotrain_crossplay.png not drawn")
        return
    side, fee = str(evaluation["side"]), float(evaluation["fee"])
    mm = {n: evaluation["eval_mm_pl"][i] for i, n in enumerate(names)}
    tr = {n: evaluation["eval_trader_pl"][i] for i, n in enumerate(names)}
    n_paths = mm[names[0]].size

    def cell(name):
        return (f"MM {mm[name].mean() * BP:+.2f} bp\n"
                f"trader {tr[name].mean() * 100:+.4f}%")

    fig, (grid, gains) = plt.subplots(1, 2, figsize=(9.5, 3.0),
                                      gridspec_kw={"width_ratios": [1, 1.4]})
    grid.axis("off")
    table = grid.table(
        cellText=[[cell("co MM v co trader"), cell("co MM v P1 trader")],
                  [cell("P1 MM v co trader"), cell("P1 MM v P1 trader")]],
        rowLabels=["co-trained MM", "Phase 1 MM"],
        colLabels=["co-trained trader", "Phase 1 trader"],
        cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 2.4)
    grid.set_title(f"cross-play, {n_paths:,} shared paths", fontsize=9)

    gains.axis("off")
    rows = []
    for name, gain, se in zip(evaluation["cross_names"], evaluation["cross_gain"],
                              evaluation["cross_2se"]):
        name = str(name)
        unit, label = (BP, "bp") if "MM gain" in name else (100, "%")
        verdict = ("no gain" if gain <= se else "GAINS")
        rows.append([name, f"{gain * unit:+.3f} +/- {se * unit:.3f} {label}",
                     verdict])
    table = gains.table(cellText=rows, colLabels=["deviation", "gain (paired, 2 SE)",
                                                  "read"],
                        cellLoc="left", loc="center",
                        colWidths=[0.62, 0.26, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.8)
    gains.set_title("a best response gains nothing: every line should read "
                    "'no gain'", fontsize=8)
    fig.suptitle(f"side {side}, MM fee {fee}: is the co-trained pair the "
                 f"equilibrium? (Phase 1 trader {str(evaluation['phase1_fingerprint'])})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "cotrain_crossplay.png", dpi=150)
    plt.close(fig)


def _spec_from(evaluation):
    #the card as the RUN saw it, rebuilt from the artefact, not today's CARD
    L, T, A, B = (float(x) for x in evaluation["card"])
    return GameSpec(L=L, T=T, A=A, B=B, side=str(evaluation["side"]),
                    rounds=int(evaluation["record_quoted_price"].shape[1]),
                    params=GameParams(a0=float(evaluation["a0"]),
                                      sd=float(evaluation["sd"]),
                                      bdc_fee=float(evaluation["trader_bdc_fee"])))


def plot_pair_views(run, evaluation):
    """The separate pipelines' own figures, for the co-trained pair, drawn by
    the separate pipelines' own code: set_path.png and
    representative_path.png via trader_diagnostics (v2.plot_game underneath,
    with the LEARNED MM's break-even on the price panel), and mm_level.png
    via mm_diagnostics.plot_level, straight off cotrain_eval.npz."""
    side, fee = str(evaluation["side"]), float(evaluation["fee"])
    spec = _spec_from(evaluation)
    if fee != 0.0:
        # at any positive fee the co-trained pair does not trade, so a v2
        # single-game figure is one rejected quote and a Bureau dump --
        # nothing to see. mm_level.png still carries the level and repay
        mm_diagnostics.plot_level(run, evaluation)
        print(f"  {run.name}: mm_level.png (path figures skipped: fee "
              f"{fee:g} > 0, the pair does not trade)")
        return
    if "set_record_quoted_price" not in evaluation:
        print(f"  {run.name}: no set-path record in cotrain_eval.npz -- "
              f"re-run cotrain.py with EVAL_ONLY = True to add it; skipping "
              f"the pair-view figures")
        return
    set_log = {key[len("set_record_"):]: value for key, value in
               evaluation.items() if key.startswith("set_record_")}
    trader_diagnostics._path_figure(
        spec, set_log, float(evaluation["set_trader_pl"]), 0, fee,
        f"side {side}, co-trained pair: the set path "
        f"({', '.join(f'{r:g}' for r in evaluation['set_path'])})",
        run / "set_path.png")
    log = {key[len("record_"):]: value for key, value in evaluation.items()
           if key.startswith("record_") and not key.startswith("record_trader_pl")
           and not key.startswith("record_mm_pl")}
    trader_pl = evaluation["record_trader_pl"]
    i = int(np.argsort(trader_pl)[trader_pl.size // 2])
    trader_diagnostics._path_figure(
        spec, log, float(trader_pl[i]), i, fee,
        f"side {side}, co-trained pair: representative game (median trader "
        f"P/L, path #{i} of {trader_pl.size:,})",
        run / "representative_path.png")
    mm_diagnostics.plot_level(run, evaluation)
    print(f"  {run.name}: set_path.png, representative_path.png, mm_level.png")


def write_summary(run, curve, evaluation):
    """cotrain_summary.txt: the run's headline numbers, from the artefacts
    alone. Old artefacts that lack a key get a pointer, not a crash."""
    E = evaluation
    side, fee = str(E["side"]), float(E["fee"])
    L, T, A, B = (float(x) for x in E["card"])
    n_rounds = (int(E["rounds"]) if "rounds" in E
                else int(E["record_quoted_price"].shape[1]))
    k = int(E["k"]) if "k" in E else None
    names = [str(n) for n in E["eval_names"]]
    mm = {n: E["eval_mm_pl"][i] for i, n in enumerate(names)}
    tr = {n: E["eval_trader_pl"][i] for i, n in enumerate(names)}
    n_paths = mm[names[0]].size
    co_mm, co_tr = mm["co MM v co trader"], tr["co MM v co trader"]

    def pct(x):
        return f"{float(np.mean(x)) * 100:+.4f}%"

    def bp(x):
        return f"{float(np.mean(x)) * BP:+.2f} bp"

    def two_se(x):
        return 2 * float(np.std(x, ddof=1)) / np.sqrt(np.size(x))

    def paired(later, earlier):
        d = np.asarray(later) - np.asarray(earlier)
        return float(d.mean()), two_se(d)

    n_blocks = len(curve["mm_pl"])
    stopped = int(curve["stopped_at"])
    lines = [
        f" SUMMARY  |  co-trained pair, side {side}, MM fee {fee:g}, "
        f"{n_rounds} rounds" + (f", K {k}" if k is not None else "")
        + f", {n_paths:,} evaluation paths",
        f"   card             L {L:,.0f}  T {T:,.0f}  A {A:g}  B {B:g}   "
        f"a0 {float(E['a0']):g}  sd {float(E['sd']):g}",
        f"   fees             trader BdC {float(E['trader_bdc_fee']):g}   "
        f"MM {fee:g} (its own Bureau cost)",
        f"   training         {int(E['n_outer'])} outer x {int(E['n_inner'])} "
        f"iterations x {int(E['batch']):,} games per agent, "
        f"{str(E['first'])} first, MM level init {str(E['mm_level_init'])}",
        f"                    trader LR {float(E['lr'][0]):g}"
        f"{' + plateau decay' if bool(E['trader_plateau']) else ' constant'}, "
        f"MM LR {float(E['lr'][1]):g}",
        f"   blocks           {n_blocks} measured (0 = as initialised);  "
        + (f"flat: stopped at outer step {stopped}" if stopped >= 0
           else f"ran all {int(E['n_outer'])} outer steps, not flat by the "
                f"paired max(2 SE, tolerance) test"),
        "",
        f" HARD GAME  |  {n_paths:,} fresh paths, every comparison paired",
        f"   MM P/L           {bp(co_mm)} +/- {two_se(co_mm) * BP:.2f}   of the "
        f"game's capital (GBP {float(E['capital_pounds']):,.0f})",
        f"   trader P/L       {pct(co_tr)} +/- {two_se(co_tr) * 100:.4f}%   "
        f"(raw: this bar is the rate path, not the policy)",
    ]
    if "eval_bdc_pl" in E:
        edge, edge_2se = paired(co_tr, E["eval_bdc_pl"])
        lines += [
            f"   always-BdC       {pct(E['eval_bdc_pl'])}   on the same paths   "
            f"(analytic {float(E['bdc_floor']) * 100:+.4f}%, check 5)",
            f"   trader edge      {edge * 100:+.4f}% +/- {edge_2se * 100:.4f}%   "
            f"over always-BdC, paired",
        ]
    else:
        lines.append("   always-BdC       not in this artefact: re-run cotrain.py "
                     "with EVAL_ONLY = True")
    for j, name in enumerate(str(n) for n in E["rule_names"]):
        gap, gap_2se = paired(co_mm, E["eval_rule_mm_pl"][j])
        lines.append(f"   vs {name:<15} MM {bp(E['eval_rule_mm_pl'][j]):>10}   "
                     f"co MM - rule {gap * BP:+.2f} +/- {gap_2se * BP:.2f} bp   "
                     f"trader vs this rule {pct(E['eval_rule_trader_pl'][j])}")
    benchmark = str(E["benchmark_name"])
    if "filled" in E:
        lines.append(f"   verdicts         agree with {benchmark} on "
                     f"{float(E['agree']) * 100:.1f}% of offered volume;  "
                     f"filled {float(E['filled']) * 100:.1f}% "
                     f"({benchmark} {float(E['filled_benchmark']) * 100:.1f}%)")
    else:
        offered = E["record_offered_amount"]
        filled = ((offered * E["record_fill_weight"]).sum()
                  / max(offered.sum(), 1e-300))
        lines.append(f"   filled           {filled * 100:.1f}% of offered "
                     f"volume (record replay; benchmark rule {benchmark})")
    lines += [
        "   threshold P/X    " + "  ".join(
            f"r{r + 1} {float(t):.4f}" for r, t in enumerate(E["threshold_by_round"]))
        + "   (fitted to the verdicts; nan = no volume)",
        "   repay on short   " + "  ".join(
            f"r{r + 1} {float(t):.3f}" for r, t in enumerate(E["repay_by_round"])),
        f"   round-1 quote    {float(np.median(E['record_quoted_price'][:, 0])):.4f} "
        f"$/GBP (median)"
        + (f"   Phase 1 trader "
           f"{float(np.median(E['p1_record_quoted_price'][:, 0])):.4f}"
           if "p1_record_quoted_price" in E else ""),
    ]
    if "set_trader_pl" in E:
        lines.append(
            f"   set path         {', '.join(f'{r:g}' for r in E['set_path'])}"
            f"  ->  MM {float(E['set_mm_pl']) * BP:+.2f} bp   trader "
            f"{float(E['set_trader_pl']) * 100:+.3f}%"
            + (f"   (always-BdC on it {float(E['set_bdc_pl']) * 100:+.3f}%)"
               if "set_bdc_pl" in E else ""))
    lines.append("")
    if len(names) >= 4:
        lines += [f" CROSS-PLAY  |  Phase 1's pair (trader "
                  f"{str(E['phase1_fingerprint'])}), same {n_paths:,} paths"]
        for name in ("P1 MM v P1 trader", "co MM v co trader",
                     "P1 MM v co trader", "co MM v P1 trader"):
            lines.append(f"   {name:<20} MM {bp(mm[name]):>10}   trader "
                         f"{pct(tr[name])}")
        lines.append("   deviation gains (paired; a best response gains "
                     "nothing, so each should be <= 0 within 2 SE):")
        for name, gain, se in zip(E["cross_names"], E["cross_gain"],
                                  E["cross_2se"]):
            name, gain, se = str(name), float(gain), float(se)
            unit, label = (BP, "bp") if "MM gain" in name else (100, "%")
            lines.append(f"     {name:<52} {gain * unit:+.3f} +/- "
                         f"{se * unit:.3f} {label}   "
                         f"{'no gain' if gain <= se else 'GAINS'}")
    else:
        lines.append(" CROSS-PLAY  |  none: cotrain ran without Phase 1's "
                     "artefacts for this side and fee")
    lines += [
        "",
        "   no DP row: the exact solution is derived for an MM that accepts at",
        "   the true rate. Against a learned MM it is neither a ceiling nor a",
        "   floor, so it is not drawn here.",
    ]
    (run / "cotrain_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    for side in SIDES:
        run, curve, evaluation = load(side)
        plot_curve(run, curve, evaluation)
        plot_policy(run, curve, evaluation)
        plot_crossplay(run, evaluation)
        plot_pair_views(run, evaluation)
        write_summary(run, curve, evaluation)
        print(f"side {side}, MM fee {MM_FEE:g}: figures and cotrain_summary.txt "
              f"in {run}/")


if __name__ == "__main__":
    main()