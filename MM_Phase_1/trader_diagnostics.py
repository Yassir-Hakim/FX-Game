import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "NN_one_offer_game"
assert (_CORE / "rl_env.py").exists(), f"core folder not found: {_CORE}"
sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(_ROOT))
"""
  trader_diagnostics.py -- the figures for the FROZEN traders, drawn ONLY
  from what train_traders.py saved (curve.npz, trader_eval.npz). Nothing is
  recomputed here and nothing is plotted anywhere else. Two kinds of figure:

    trader_curve_behaviour.png   one per trader_{side}_g{fee}/ folder, in
                                 that folder: the learning curve against the
                                 always-Bureau floor, and what the frozen
                                 trader DOES -- per-round quote over the
                                 true rate against the MM's break-even, and
                                 the share of offered volume filled. A
                                 trader that has withdrawn shows as quotes
                                 far from the break-even and fills at zero.
    trader_sweep_{side}.png      in Results/mm_phase1/, when two or more
                                 fees exist for a side: trader P/L (with the
                                 Bureau floor), fill share and round-1 quote
                                 against MM_FEE. The fee at which the fills
                                 die is the fee at which the market dies.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Mechanics.fx_mechanics import results_path

# ============================ SETTINGS ======================================
SIDES = ("A", "B")      # which sides to plot; every trader_{side}_g*/ folder
                        # holding a trader_eval.npz is picked up per side
# ============================================================================


def runs_for(side):
    root = results_path("mm_phase1")
    found = []
    for folder in sorted(root.glob(f"trader_{side}_g*")):
        if (folder / "trader_eval.npz").exists():
            found.append(folder)
    return found


def load(folder):
    evaluation = np.load(folder / "trader_eval.npz", allow_pickle=False)
    evaluation = {key: evaluation[key] for key in evaluation.files}
    curve = np.load(folder / "curve.npz", allow_pickle=False)
    return evaluation, {key: curve[key] for key in curve.files}


def fill_share_by_round(evaluation):
    offered = evaluation["record_offered_amount"]
    filled = offered * evaluation["record_fill_weight"]
    total = offered.sum(axis=0)
    return np.where(total > 0, filled.sum(axis=0) / np.maximum(total, 1e-300),
                    0.0), total.sum() and filled.sum() / offered.sum()


def plot_run(folder):
    evaluation, curve = load(folder)
    side, fee = str(evaluation["side"]), float(evaluation["mm_fee"])
    rounds = int(evaluation["rounds"])
    pl = evaluation["pl"]
    two_se = 2 * pl.std() / np.sqrt(pl.size)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6.6))

    # ---- the learning curve, floored --------------------------------------
    top.plot(curve["iters"], curve["hard_pl"] * 100, color="k", lw=1.4,
             label="hard-game validation mean")
    top.axhline(float(evaluation["bdc_floor"]) * 100, color="tab:orange",
                lw=1.1, ls="--",
                label=f"always-Bureau floor (DP) "
                      f"{float(evaluation['bdc_floor']) * 100:+.3f}%")
    top.axhline(pl.mean() * 100, color="0.5", lw=0.9, ls=":",
                label=f"final, {int(evaluation['eval_paths']):,} paths: "
                      f"{pl.mean() * 100:+.4f}% +/- {two_se * 100:.4f}%")
    top.set_xlabel("iteration")
    top.set_ylabel("trader P/L, % of L")
    top.set_title(f"side {side}, MM fee {fee:g}, trader Bureau fee "
                  f"{float(evaluation['bdc_fee']):g}: the frozen trader")
    top.grid(alpha=0.3)
    top.legend(fontsize=7, loc="best")

    # ---- what it does: quotes against the break-even, and the fills -------
    price_over_rate = (evaluation["record_quoted_price"]
                       / evaluation["record_hidden_rate"])
    level_over_rate = (evaluation["record_break_even"]
                       / evaluation["record_hidden_rate"])
    x = np.arange(1, rounds + 1)
    quartiles = np.percentile(price_over_rate, [25, 50, 75], axis=0)
    bottom.plot(x, quartiles[1], color="k", lw=1.4, marker="o", ms=4,
                label="quote / true rate (median, IQR band)")
    bottom.fill_between(x, quartiles[0], quartiles[2], color="k", alpha=0.12,
                        lw=0)
    bottom.plot(x, np.median(level_over_rate, axis=0), color="tab:red",
                lw=1.1, label=f"the MM's break-even ({str(evaluation['rule'])})")
    share, overall = fill_share_by_round(evaluation)
    twin = bottom.twinx()
    twin.bar(x, share * 100, width=0.25, color="tab:blue", alpha=0.5,
             label=f"filled, % of offered volume (overall {overall * 100:.2f}%)")
    twin.set_ylabel("filled, % of offered volume", color="tab:blue",
                    fontsize=8)
    twin.set_ylim(0, max(102, share.max() * 120))
    twin.tick_params(axis="y", labelsize=7, colors="tab:blue")
    bottom.set_xlabel("round")
    bottom.set_xticks(x)
    bottom.set_ylabel("price / true rate")
    bottom.grid(alpha=0.3)
    lines = bottom.get_lines() + [twin.containers[0]]
    bottom.legend(lines, [line.get_label() for line in lines], fontsize=7,
                  loc="best")
    fig.tight_layout()
    fig.savefig(folder / "trader_curve_behaviour.png", dpi=150)
    plt.close(fig)


def plot_sweep(side, folders):
    rows = []
    for folder in folders:
        evaluation, _ = load(folder)
        pl = evaluation["pl"]
        _, overall = fill_share_by_round(evaluation)
        rows.append(dict(
            fee=float(evaluation["mm_fee"]), pl=pl.mean(),
            two_se=2 * pl.std() / np.sqrt(pl.size), filled=overall,
            offer=float(np.median(evaluation["record_quoted_price"][:, 0]))
                  / float(evaluation["a0"]),
            floor=float(evaluation["bdc_floor"])))
    rows.sort(key=lambda row: row["fee"])
    fee = np.array([row["fee"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.8), sharex=True)
    value, volume, quote = axes
    value.errorbar(fee, [row["pl"] * 100 for row in rows],
                   yerr=[row["two_se"] * 100 for row in rows], color="k",
                   lw=1.4, marker="o", ms=4, capsize=3,
                   label="frozen trader, hard game (2 SE)")
    value.axhline(rows[0]["floor"] * 100, color="tab:orange", lw=1.1, ls="--",
                  label="always-Bureau floor (DP)")
    value.set_ylabel("trader P/L, % of L")
    value.set_title(f"side {side}: the trader's best response as the MM's "
                    f"own Bureau fee rises")
    value.grid(alpha=0.3)
    value.legend(fontsize=7, loc="best")
    volume.plot(fee, [row["filled"] * 100 for row in rows], color="k", lw=1.4,
                marker="o", ms=4)
    volume.set_ylabel("filled, % of offered volume")
    volume.grid(alpha=0.3)
    quote.plot(fee, [row["offer"] for row in rows], color="k", lw=1.4,
               marker="o", ms=4)
    quote.set_ylabel("round-1 quote / a0")
    quote.set_xlabel("MM fee (the MM's own Bureau cost)")
    quote.grid(alpha=0.3)
    fig.tight_layout()
    out = results_path("mm_phase1") / f"trader_sweep_{side}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# the report for a game WITH an MM fee -- rl_diagnostics.report()'s stand-in
# ---------------------------------------------------------------------------
SET_PATH = [1.30, 1.36, 1.37, 1.41, 1.33]   # the demo game's rates, from
                                            # BdC_Calculations.ods (Game
                                            # sheet): rounds 1-4 then settle


def _forced_moves(spec, rates):
    #the z-draws that put the hidden rate exactly on these values
    a = np.asarray(rates, dtype=float)
    previous = np.concatenate(([spec.params.a0], a[:-1]))
    return torch.tensor((a - previous) / spec.params.sd).unsqueeze(0)


def _trace_from_record(spec, log, i, pl_value):
    """One game of a fee-game record -> v2's own trace dict, the same shape
    rl_diagnostics._eps_to_trace builds, so v2.plot_game draws these figures
    with exactly the code that draws every other path figure in the project.
    v2's traces are ROLE-ordered: 'c' is the initial currency draining and
    'd' is the target banked, on BOTH sides -- so held IS c and banked IS d,
    and reordering by side here would swap the two curves."""
    import NN_one_offer_game.rl_diagnostics as D
    from Mechanics.fx_mechanics import trade_rate
    sd, anchor, greed = spec.params.sd, spec.params.a0, D._greed(spec)
    held_after_bureau = log["trader_held_after_bureau"][i]
    banked_after_bureau = log["trader_banked_after_bureau"][i]
    held_after_fill = log["trader_held_after_fill"][i]
    banked_after_fill = log["trader_banked_after_fill"][i]
    rounds = []
    for n in range(1, spec.rounds + 1):
        j = n - 1
        quote = float(log["quoted_price"][i, j])
        offered = float(log["offered_amount"][i, j])
        offers = []
        if offered > 1e-6:
            offers.append({"offer_no": 1, "z": (quote - anchor) / sd * greed,
                           "P": trade_rate(quote, spec.side), "quote": quote,
                           "q": offered,
                           "accepted": bool(log["fill_weight"][i, j] > 0.5),
                           "c": float(held_after_fill[j]),
                           "d": float(banked_after_fill[j])})
        rounds.append({"n": n, "anchor": anchor,
                       "X": float(log["hidden_rate"][i, j]), "offers": offers,
                       "bdc_dump": float(held_after_fill[j]
                                         - held_after_bureau[j]),
                       "c": float(held_after_bureau[j]),
                       "d": float(banked_after_bureau[j])})
        anchor = float(log["hidden_rate"][i, j])
    return {"spec": spec, "rounds": rounds,
            "a5": float(log["settlement_rate"][i]),
            "c": float(held_after_bureau[-1]), "d": float(banked_after_bureau[-1]),
            "W": spec.L * (1 + pl_value), "pl": pl_value}


def _path_figure(spec, log, pl_value, i, mm_fee, title, out_path):
    """One game, drawn by v2.plot_game -- the same function rl_diagnostics
    uses -- so the figure matches the others exactly. Two panels rather than
    four: there is no DP row, because the exact solution has not been derived
    for a fee-charging MM."""
    from DP_Models.v2_multiple_rounds import plot_game
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
    plot_game(_trace_from_record(spec, log, i, pl_value), ax_rate=axes[0],
              ax_inv=axes[1],
              title=f"the trained trader, P/L = {pl_value * 100:+.2f}%")
    # plot_game defaults its title to "Optimal play" -- true of the DP, not of
    # a learner, and doubly wrong here where no DP exists for the fee game
    figure.suptitle(f"{title}   MM fee {mm_fee:g}", y=0.99, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def report_fee(spec, net, curve, out_dir, mm_fee, n_paths=20_000, seed=4242):
    """The standard outputs for a game whose MM charges itself a fee, where
    rl_diagnostics.report() CANNOT be used: it drives the policy through the
    certified env and solves the v2 DP against it, and both have an MM that
    accepts at the true rate. Everything here is replayed through mm_train's
    fee-aware rule instead, which check 1F certifies equal to
    torch_train_fee.rollout at this fee, so every number describes the game
    that was actually trained. There is no DP line: the exact solution for
    the fee game has not been derived. The always-Bureau floor is the
    reference in its place."""
    import mm_train as M
    from DP_Models.v2_multiple_rounds import bdc_baseline
    out_dir.mkdir(parents=True, exist_ok=True)
    rule = next(r for r in M.reference_rules(mm_fee)
                if r.name == ("fee-aware+flatten" if mm_fee > 0
                              else "naive+flatten"))
    generator = torch.Generator().manual_seed(seed)
    moves = torch.randn(n_paths, spec.rounds + 1, generator=generator)
    net.eval()
    with torch.no_grad():
        _, pl, log = M.rollout(None, net, spec, moves, rule=rule, fee=mm_fee,
                               record=True)
        _, set_pl, set_log = M.rollout(None, net, spec,
                                       _forced_moves(spec, SET_PATH),
                                       rule=rule, fee=mm_fee, record=True)
    pl = pl.numpy()
    two_se = 2 * pl.std() / np.sqrt(pl.size)
    try:
        floor = float(bdc_baseline(spec)[1])
    except Exception:
        floor = float("nan")
    filled = ((log["offered_amount"] * log["fill_weight"]).sum()
              / max(log["offered_amount"].sum(), 1e-300))

    # ---- learning curve: the floor in the DP's place ----------------------
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    axes.plot(curve[0], np.array(curve[1]) * 100, color="k", lw=1.5,
              label="hard-game validation mean")
    if np.isfinite(floor):
        axes.axhline(floor * 100, ls="--", color="tab:orange", lw=1.4,
                     label=f"always-Bureau floor ({floor * 100:+.3f}%)")
    axes.axhline(pl.mean() * 100, ls=":", color="0.5", lw=1.1,
                 label=f"final, {n_paths:,} paths ({pl.mean() * 100:+.4f}%)")
    axes.set_xlabel("iteration")
    axes.set_ylabel("trader P/L, % of L")
    axes.set_title(f"side {spec.side}, MM fee {mm_fee:g}: learning curve "
                   f"(no DP: not derived for the fee game)")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "learning_curve.png", dpi=150)
    plt.close(figure)

    # ---- P/L histogram ----------------------------------------------------
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.hist(pl * 100, bins=80, color="tab:blue", alpha=0.75)
    axes.axvline(pl.mean() * 100, color="k", lw=1.4,
                 label=f"mean {pl.mean() * 100:+.4f}%")
    if np.isfinite(floor):
        axes.axvline(floor * 100, color="tab:orange", ls="--", lw=1.4,
                     label=f"always-Bureau floor {floor * 100:+.3f}%")
    axes.set_xlabel("P/L, % of L")
    axes.set_ylabel("games")
    axes.set_title(f"side {spec.side}, MM fee {mm_fee:g}: hard game, "
                   f"{n_paths:,} paths")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "pl_hist.png", dpi=150)
    plt.close(figure)

    # ---- what the policy does, per round ----------------------------------
    rounds = np.arange(1, spec.rounds + 1)
    figure, (offers, volume) = plt.subplots(2, 1, figsize=(7.0, 6.0),
                                            sharex=True)
    quartiles = np.percentile(log["quoted_price"] / log["hidden_rate"],
                              [25, 50, 75], axis=0)
    offers.plot(rounds, quartiles[1], color="k", lw=1.5, marker="o", ms=4,
                label="quote / true rate (median, IQR)")
    offers.fill_between(rounds, quartiles[0], quartiles[2], color="k",
                        alpha=0.12, lw=0)
    offers.plot(rounds, np.median(log["break_even"] / log["hidden_rate"],
                                  axis=0), color="tab:red", lw=1.2,
                label=f"the MM's break-even ({rule.name})")
    offers.plot(rounds, set_log["quoted_price"][0] / set_log["hidden_rate"][0],
                color="tab:green", lw=1.2, ls="--", marker="s", ms=4,
                label=f"the set path ({', '.join(f'{r:g}' for r in SET_PATH)}), "
                      f"P/L {float(set_pl[0]) * 100:+.3f}%")
    offers.set_ylabel("price / true rate")
    offers.set_title(f"side {spec.side}, MM fee {mm_fee:g}: the trained policy")
    offers.grid(alpha=0.3)
    offers.legend(fontsize=7.5)
    offered = log["offered_amount"]
    share = np.where(offered.sum(axis=0) > 0,
                     (offered * log["fill_weight"]).sum(axis=0)
                     / np.maximum(offered.sum(axis=0), 1e-300), 0.0)
    volume.bar(rounds - 0.16, offered.mean(axis=0) / spec.L * 100, width=0.32,
               color="0.6", label="offered, % of L (mean)")
    volume.bar(rounds + 0.16, share * 100, width=0.32, color="tab:blue",
               label=f"filled, % of offered (overall {filled * 100:.2f}%)")
    volume.set_xlabel("round")
    volume.set_xticks(rounds)
    volume.set_ylabel("%")
    volume.grid(alpha=0.3)
    volume.legend(fontsize=7.5)
    figure.tight_layout()
    figure.savefig(out_dir / "policy_by_round.png", dpi=150)
    plt.close(figure)

    # ---- two single-game figures: the set path, and a representative one --
    _path_figure(spec, set_log, float(set_pl[0]), 0, mm_fee,
                 f"side {spec.side}: the set path "
                 f"({', '.join(f'{r:g}' for r in SET_PATH)})",
                 out_dir / "set_path.png")
    representative = int(np.argsort(pl)[len(pl) // 2])   # as plot_path picks it
    _path_figure(spec, log, float(pl[representative]), representative, mm_fee,
                 f"side {spec.side}: representative game (median P/L, "
                 f"path #{representative} of {n_paths:,})",
                 out_dir / "representative_path.png")

    # ---- summary ----------------------------------------------------------
    lines = [
        f" SUMMARY  |  side {spec.side}, rounds {spec.rounds}, K {spec.K}, "
        f"{n_paths:,} paths",
        f"   card             L {spec.L:,.0f}  T {spec.T:,.0f}  A {spec.A}  "
        f"B {spec.B}",
        f"   MM fee           {mm_fee:g}   (its own Bureau cost; the trader "
        f"never sees it)",
        f"   trader BdC fee   {spec.params.bdc_fee:g}",
        f"   hard-game P/L    {pl.mean() * 100:+.4f}% +/- {two_se * 100:.4f}%",
        f"   always-Bureau    {floor * 100:+.4f}%   (the outside option)",
        f"   edge over floor  {(pl.mean() - floor) * 100:+.4f}%",
        f"   filled           {filled * 100:.2f}% of offered volume",
        f"   round-1 quote/a0 "
        f"{float(np.median(log['quoted_price'][:, 0])) / spec.params.a0:.4f}",
        f"   set path         {', '.join(f'{r:g}' for r in SET_PATH)}  ->  "
        f"P/L {float(set_pl[0]) * 100:+.3f}%",
        "",
        "   no DP line: the exact solution has been derived for an MM that",
        "   accepts at the true rate, not for one that charges itself a fee.",
        "   Replayed through mm_train's " + rule.name + " rule, which check 1F",
        "   certifies equal to torch_train_fee.rollout at this fee.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nFILES: learning_curve.png, pl_hist.png, policy_by_round.png, "
          f"set_path.png, representative_path.png, summary.txt"
          f"\n\nsix outputs in {out_dir}/", flush=True)


def main():
    for side in SIDES:
        folders = runs_for(side)
        if not folders:
            print(f"side {side}: no trader_{side}_g*/trader_eval.npz found -- "
                  f"run train_traders.py first")
            continue
        for folder in folders:
            plot_run(folder)
            print(f"side {side}: {folder.name}/trader_curve_behaviour.png")
        if len(folders) >= 2:
            print(f"side {side}: {plot_sweep(side, folders)}")
        else:
            print(f"side {side}: one fee only, no sweep figure (train at "
                  f"another T.MM_FEE to get one)")


if __name__ == "__main__":
    main()