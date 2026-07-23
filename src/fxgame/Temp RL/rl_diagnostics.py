"""
Evaluation + plots for a trained agent on the v2 RL game.

One entry point, run_diagnostics(model, run_dir, env_factory), which:
  * plays the deterministic policy many times (Monte Carlo -- the only way to
    score a learned net; it has no closed form),
  * writes three figures to run_dir:
        diagnostics_path.png   v2-style per-game path (offers, BdC, inventory)
        learning_curve.png     eval P/L vs timesteps (from evaluations.npz)
        pl_histogram.png       distribution of eval P/L
  * writes summary.txt (mean +/- 2 s.e., median, P(target hit), terminals),
  * returns the summary as a dict so train.py can print the scorecard.

Everything here reads the raw env (reward scaling never touches observations,
so a policy trained on scaled reward plays identically on the raw env).
"""

from __future__ import annotations

import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from v2rl_env import V2Game, GameSpec, Z_OFFER

# reference lines (both closed-form, so we quote them rather than simulate them)
BDC_BASELINE_PCT = -1.444    # analytic always-BdC value on the T1 card
V2_DP_PCT = -0.356           # v2's exact DP optimum


def rollout(model, env, seed=None):
    """Play one deterministic episode; return (flat event list, P/L).

    Each event records the step, round, phase, the offer/BdC price, the hidden
    rate (peeked for the plot only), the pounds/dollars before and after, and
    -- for offers -- whether it cleared.
    """
    obs, _ = env.reset(seed=seed)
    events = []
    done, r, step = False, 0.0, 0
    while not done:
        step += 1
        phase, rnd = env.phase, env.n
        anchor, hidden = env.a, env.X
        c0, d0 = env.c, env.d
        raw, _ = model.predict(obs, deterministic=True)
        # a discrete (DQN) env decodes its integer action to [lever, fraction];
        # a continuous (PPO) env is already in that form.
        if hasattr(env, "decode_action"):
            cont, step_action = env.decode_action(int(raw)), int(raw)
        else:
            cont, step_action = np.asarray(raw, np.float32), raw
        if phase == 0:  # offer
            z = float(np.clip(cont[0], -Z_OFFER, Z_OFFER))   # price in sds from anchor
            price = anchor + env.sd * z
            qty = float(np.clip(cont[1], 0.0, 1.0)) * c0
            accepted = price < hidden
            obs, r, term, trunc, _ = env.step(step_action)
            events.append(dict(step=step, round=rnd, phase="offer", price=price,
                               hidden_rate=hidden, quantity=qty, accepted=accepted,
                               pounds_before=c0, pounds_after=env.c,
                               dollars_before=d0, dollars_after=env.d))
        else:  # BdC
            qty = float(np.clip(cont[1], 0.0, 1.0)) * c0
            price = (1.0 - env.f) * hidden
            obs, r, term, trunc, _ = env.step(step_action)
            events.append(dict(step=step, round=rnd, phase="bdc", price=price,
                               hidden_rate=hidden, quantity=qty, accepted=None,
                               pounds_before=c0, pounds_after=env.c,
                               dollars_before=d0, dollars_after=env.d))
        done = term or trunc
    return events, r


def evaluate(model, env_factory, n_games=2000, seed=123):
    """Monte Carlo over deterministic play. Returns per-game arrays plus the
    trace of the representative (median-P/L) game for plotting."""
    env = env_factory()
    pls = np.empty(n_games)
    terminal_pounds = np.empty(n_games)
    terminal_dollars = np.empty(n_games)
    traces = []
    for i in range(n_games):
        events, pl = rollout(model, env, seed=seed + i)
        pls[i] = pl
        terminal_pounds[i] = events[-1]["pounds_after"]
        terminal_dollars[i] = events[-1]["dollars_after"]
        traces.append(events)
    rep = int(np.argmin(np.abs(pls - np.median(pls))))
    return {"pl": pls, "pounds": terminal_pounds, "dollars": terminal_dollars,
            "rep_events": traces[rep], "rep_pl": float(pls[rep])}


# --------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------- #
def plot_policy_path(events, pl, save_path, dollar_target, algorithm="PPO"):
    """v2-style two panels: rate/offers on top, inventory on the bottom."""
    fig, (ax_rate, ax_inv) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    seen = set()
    for e in events:
        x = e["step"]
        ax_rate.plot([x - 0.45, x + 0.45], [e["hidden_rate"]] * 2, color="black", lw=1.4)
        if e["phase"] == "offer":
            acc = bool(e["accepted"])
            lab = "offer accepted" if acc else "offer rejected"
            size = 30 + 170 * e["quantity"] / max(e["pounds_before"], 1.0)
            ax_rate.scatter(x, e["price"], s=size,
                            facecolors="tab:green" if acc else "none",
                            edgecolors="tab:green" if acc else "tab:red",
                            label=lab if lab not in seen else None, zorder=3)
            seen.add(lab)
        elif e["phase"] == "bdc" and e["quantity"] > 1e-6:
            lab = "BdC sale"
            ax_rate.scatter(x, e["price"], marker="D", color="tab:blue", s=55,
                            label=lab if lab not in seen else None, zorder=3)
            seen.add(lab)
    for rnd in sorted({e["round"] for e in events}):
        steps = [e["step"] for e in events if e["round"] == rnd]
        ax_rate.axvspan(min(steps) - 0.5, max(steps) + 0.5, color="0.5",
                        alpha=0.06 if rnd % 2 else 0.12)
        ax_rate.text(float(np.mean(steps)), ax_rate.get_ylim()[1], f"Round {rnd}",
                     ha="center", va="bottom", fontsize=9)
    steps = [0] + [e["step"] for e in events]
    pounds = [events[0]["pounds_before"]] + [e["pounds_after"] for e in events]
    dollars = [events[0]["dollars_before"]] + [e["dollars_after"] for e in events]
    ax_inv.step(steps, pounds, where="post", color="tab:blue", label="GBP remaining")
    ax_usd = ax_inv.twinx()
    ax_usd.step(steps, dollars, where="post", color="tab:orange", label="USD banked")
    ax_usd.axhline(dollar_target, color="tab:orange", ls="--", alpha=0.6, label="USD target")

    ax_rate.set(ylabel="USD per GBP",
                title=f"{algorithm} policy \u2014 representative game (P/L = {pl*100:+.2f}%)")
    if ax_rate.get_legend_handles_labels()[0]:
        ax_rate.legend(loc="best", fontsize=8)
    ax_inv.set(xlabel="offer / BdC step", ylabel="GBP remaining")
    ax_usd.set_ylabel("USD banked")
    h1, l1 = ax_inv.get_legend_handles_labels()
    h2, l2 = ax_usd.get_legend_handles_labels()
    ax_inv.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_learning_curve(npz_path, save_path, algorithm="PPO"):
    if not os.path.exists(npz_path):
        return False
    ev = np.load(npz_path)
    ts, pl = ev["timesteps"], ev["results"].mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, pl, marker="o", lw=1.5, label=f"{algorithm} (eval mean)")
    ax.axhline(BDC_BASELINE_PCT, ls=":", color="grey",
               label=f"always-BdC ({BDC_BASELINE_PCT:.2f}%)")
    ax.axhline(V2_DP_PCT, ls="--", color="tab:green",
               label=f"v2 DP optimum ({V2_DP_PCT:.2f}%)")
    ax.set(xlabel="timesteps", ylabel="P/L (%)", title=f"{algorithm} learning curve")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=160); plt.close(fig)
    return True


def plot_pl_histogram(pls_pct, mean_pct, save_path, algorithm="PPO"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pls_pct, bins=40, edgecolor="white")
    ax.axvline(mean_pct, color="black", ls="--", label=f"mean {mean_pct:+.2f}%")
    ax.axvline(V2_DP_PCT, color="tab:green", ls="--", label=f"DP optimum {V2_DP_PCT:+.2f}%")
    ax.set(xlabel="P/L (%)", ylabel="games",
           title=f"{algorithm} evaluation P/L distribution")
    ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=160); plt.close(fig)


def run_diagnostics(model, run_dir, env_factory=None, n_games=2000, seed=123,
                    algorithm="PPO"):
    """Evaluate, write the three figures + summary.txt, return the summary."""
    env_factory = env_factory or (lambda: V2Game(GameSpec()))
    os.makedirs(run_dir, exist_ok=True)
    spec = env_factory().spec
    target = spec.T

    ev = evaluate(model, env_factory, n_games=n_games, seed=seed)
    pls_pct = ev["pl"] * 100.0
    mean_pct = float(pls_pct.mean())
    se2_pct = float(2 * pls_pct.std(ddof=1) / math.sqrt(n_games))
    hit = float(np.mean(ev["dollars"] >= target))

    plot_policy_path(ev["rep_events"], ev["rep_pl"],
                     os.path.join(run_dir, "diagnostics_path.png"), target,
                     algorithm=algorithm)
    plot_learning_curve(os.path.join(run_dir, "evaluations.npz"),
                        os.path.join(run_dir, "learning_curve.png"),
                        algorithm=algorithm)
    plot_pl_histogram(pls_pct, mean_pct, os.path.join(run_dir, "pl_histogram.png"),
                      algorithm=algorithm)

    lines = [
        f"evaluation games      {n_games}",
        f"mean P/L              {mean_pct:+.3f}%  (2 s.e. {se2_pct:.3f}%)",
        f"median P/L            {float(np.median(pls_pct)):+.3f}%",
        f"P(dollar target met)  {hit*100:.1f}%",
        f"mean terminal GBP     {float(ev['pounds'].mean()):,.0f}",
        f"mean terminal USD     {float(ev['dollars'].mean()):,.0f}",
        f"reference: always-BdC {BDC_BASELINE_PCT:+.3f}%   v2 DP {V2_DP_PCT:+.3f}%",
    ]
    with open(os.path.join(run_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return {"mean_pct": mean_pct, "se2_pct": se2_pct, "median_pct": float(np.median(pls_pct)),
            "hit_pct": hit * 100, "terminal_gbp": float(ev["pounds"].mean()),
            "terminal_usd": float(ev["dollars"].mean())}