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
fee_sweep.py -- the MM's Bureau fee g swept, two arms side by side:
  v0     fee_aware_v0's closed form: one round, one all-or-nothing offer,
         target currency per unit held. Solved at every fee.
  torch  torch_train_fee unchanged, at the engine's own ROUNDS. Trained at
         every TORCH_EVERY-th fee. Its round-1 offer is read straight off the
         network at the anchor (round-1 features are constants).
The objectives differ -- v0 has no card, torch has the whole game -- and they
still land in the same place. Both arms report the starting quote and the
probability that offer is filled, the same Phi at each arm's own quote.
Torch also reports its eval P/L against the always-Bureau floor on the same
paths, which is the collapse seen from the P/L side.

UNITS. The engine quotes every price in $/GBP on both sides. Side A is shown
that way. Side B holds dollars and wants pounds, so its quote is SHOWN as
GBP/$ -- the reciprocal, display only; nothing in the arithmetic changes.
Outputs: summary.txt, fee_sweep.npz, fee_sweep.png in mm_phase1/fee_sweep/.
"""

import math
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import MM_Phase_1.torch_train_fee as T
from MM_Phase_1.fee_aware_v0 import (closed_form_optimal_rate,
                                    quote_from_threshold)

# ============================ SETTINGS ======================================
FEES = np.round(np.arange(0.0, 0.01, 0.001), 3)   # g: 21 points, 0 -> f
TORCH_EVERY =3        # train at FEES[::TORCH_EVERY]; v0 runs at every fee
SIDES = ("A", "B")
# ============================================================================

torch.set_default_dtype(torch.float64)


def shown(quote_usd_per_gbp, side):
    """Display units: side A $/GBP as the engine has it, side B GBP/$."""
    return quote_usd_per_gbp if side == "A" else 1.0 / quote_usd_per_gbp


def unit(side):
    return "$/GBP" if side == "A" else "GBP/$"


def fill_prob(quote, g, side, a0, sd):
    """P(the offer is filled), from the MM's fee-aware rule: side A needs
    X > quote/(1-g), side B needs X < quote(1-g); X ~ N(a0, sd) in round 1.
    quote in the engine's $/GBP on both sides."""
    threshold = quote / (1.0 - g) if side == "A" else quote * (1.0 - g)
    z = (threshold - a0) / sd
    return float(1.0 - norm.cdf(z)) if side == "A" else float(norm.cdf(z))


def first_offer(net, spec):
    """The trained net's round-1 quote in the engine's $/GBP. At round 1 the
    features are constants (full book, nothing banked, anchor a0, empty
    history), so the offer is one deterministic number."""
    with torch.no_grad():
        feats = T.features(1.0, 0.0, torch.full((1,), float(spec.L)),
                           torch.zeros(1),
                           torch.full((1,), float(spec.params.a0)), spec,
                           torch.zeros(1, T.LAGS) if T.LAGS else None)
        price, _ = T.squash(net(feats),
                            torch.full((1,), float(spec.params.a0)))
    return float(price[0])


def torch_arm(spec, g, outdir):
    """Train at fee g; return the round-1 quote ($/GBP), its fill probability,
    and the eval P/L paired with the always-Bureau floor on the same paths."""
    T.MM_FEE = float(g)                  # rollout reads this global
    run = outdir / f"torch_{spec.side}_g{g:g}"
    run.mkdir(parents=True, exist_ok=True)
    net, _ = T.train(spec, run)
    gen = torch.Generator().manual_seed(T.SEED_PATH + 2)   # the engine's eval seed
    moves = torch.randn(T.EVAL_PATHS, spec.rounds + 1, generator=gen)
    with torch.no_grad():
        pl = T.rollout(net, spec, moves, tau=None)
        # always-Bureau: offer nothing, dump everything, every round
        floor = T.rollout(None, spec, moves, tau=None,
                          forced=[(spec.params.a0, 0.0, 1.0)] * spec.rounds)
    quote = first_offer(net, spec)
    return dict(quote=quote,
                fill=fill_prob(quote, g, spec.side, spec.params.a0,
                               spec.params.sd),
                pl=float(pl.mean()), floor=float(floor.mean()))


def main():
    outdir = T.results_path("mm_phase1/fee_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    specs = {s: T.build_spec(s) for s in SIDES}
    a0, sd, f = (specs["A"].params.a0, specs["A"].params.sd,
                 specs["A"].params.bdc_fee)

    v0 = {s: [] for s in SIDES}
    tv = {s: {} for s in SIDES}
    for k, g in enumerate(FEES):
        for s in SIDES:
            threshold = closed_form_optimal_rate(a0, sd, f, s, float(g))
            quote = quote_from_threshold(threshold, s, float(g))
            v0[s].append(dict(quote=quote,
                              fill=fill_prob(quote, float(g), s, a0, sd)))
        if k % TORCH_EVERY == 0:
            for s in SIDES:
                t0 = time.time()
                tv[s][k] = torch_arm(specs[s], float(g), outdir)
                print(f"  torch side {s} g={g:g}: {time.time() - t0:.0f}s",
                      flush=True)

    lines = [f"MM fee g swept {FEES[0]:g} .. {FEES[-1]:g} ({len(FEES)} "
             f"points), trader BdC fee f={f:g}, a0={a0:g} $/GBP, sd={sd:g}",
             f"v0: fee_aware_v0 closed form, every fee.  torch: "
             f"torch_train_fee at ROUNDS={specs['A'].rounds}, every "
             f"{TORCH_EVERY}th fee ({T.ITERS}x{T.BATCH}, eval "
             f"{T.EVAL_PATHS:,}).  fill = P(offer filled)"]
    for s in SIDES:
        lines += ["", f"side {s}  (quotes in {unit(s)})",
                  f"{'fee':>6} {'v0 quote':>9} {'v0 fill':>9}  |"
                  f"{'t quote':>9} {'t fill':>9} {'t P/L%':>9} {'floor%':>9}"]
        for k, g in enumerate(FEES):
            v = v0[s][k]
            left = f"{g:6.3f} {shown(v['quote'], s):9.4f} {v['fill']:9.2%}"
            if k in tv[s]:
                t = tv[s][k]
                right = (f"{shown(t['quote'], s):9.4f} {t['fill']:9.2%} "
                         f"{t['pl'] * 100:+9.4f} {t['floor'] * 100:+9.4f}")
            else:
                right = f"{'--':>9} {'--':>9} {'--':>9} {'--':>9}"
            lines.append(left + "  |" + right)
    txt = "\n".join(lines)
    print("\n" + txt)
    (outdir / "summary.txt").write_text(txt + "\n")

    # ---- figure: one column per side (own units), quote above fill --------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    tk = sorted(tv["A"])
    for col, (s, colour) in enumerate(zip(SIDES, ("tab:blue", "tab:red"))):
        top, bot = axes[0, col], axes[1, col]
        top.plot(FEES, [shown(v["quote"], s) for v in v0[s]], color=colour,
                 label="v0 closed form")
        top.plot(FEES[tk], [shown(tv[s][k]["quote"], s) for k in tk], "o",
                 color=colour, label="torch, round 1")
        top.axhline(shown(a0, s), color="grey", lw=0.8, ls=":",
                    label=f"anchor {shown(a0, s):.4g} {unit(s)}")
        bot.plot(FEES, np.maximum([v["fill"] for v in v0[s]], 1e-9),
                 color=colour, label="v0 closed form")
        bot.plot(FEES[tk], np.maximum([tv[s][k]["fill"] for k in tk], 1e-9),
                 "o", color=colour, label="torch, round 1")
        for ax in (top, bot):
            ax.axvline(f, color="k", lw=0.8, ls="--")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        top.set_title(f"side {s}")
        top.set_ylabel(f"round-1 quote, {unit(s)}")
        bot.set_yscale("log")
        bot.set_ylabel("P(offer filled), floored at 1e-9")
        bot.set_xlabel("MM fee g")
    axes[1, 0].sharey(axes[1, 1])
    fig.suptitle(f"MM fee sweep: where the offer sits and how often it fills "
                 f"(no trade at g = f = {f:g})")
    fig.tight_layout()
    fig.savefig(outdir / "fee_sweep.png", dpi=150)
    plt.close(fig)

    # npz keeps the engine's $/GBP on both sides; convert on read if needed
    arrs = dict(fees=FEES, a0=a0, sd=sd, bdc_fee=f, rounds=specs["A"].rounds,
                quote_units="USD per GBP, both sides")
    for s in SIDES:
        p = s.lower()
        for name in ("quote", "fill"):
            arrs[f"{p}_v0_{name}"] = np.array([v[name] for v in v0[s]])
        for name in ("quote", "fill", "pl", "floor"):
            arrs[f"{p}_t_{name}"] = np.array(
                [tv[s][k][name] if k in tv[s] else math.nan
                 for k in range(len(FEES))])
    np.savez(outdir / "fee_sweep.npz", **arrs)
    print(f"\nartefacts in {outdir}/")


if __name__ == "__main__":
    main()