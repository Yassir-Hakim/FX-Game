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
  gen_train.py -- PHASE 2.1: the trader trained in the wider world.
  torch_train's trader -- the same network, objective and rollout, against
  the same hardcoded market maker (accept iff the trade is good at the true
  rate) -- but no longer trained on the card's one market. Every block of
  games draws its own opening rate a0 and volatility sd from a window, so the
  trader learns a strategy for a family of markets and is then let loose on
  the card 

THE SCHEME (Avi's, meeting 3, verbatim in the numbers): each iteration draws
  torch_train's own 1,000 paths and cuts them into BLOCKS = 5 blocks of 200;
  each block gets one (a0, sd) pair, uniform and independent from the
  windows, as the CARD with those two numbers swapped in. torch_train.rollout
  then plays each block on its own market exactly as it plays the card, and
  the batch loss is the mean over the blocks. 

PATHS THAT TOUCH ZERO are dropped from the loss and counted, per Avi 

DISTRIBUTION KNOWLEDGE: none reaches the learner. The windows below are what
  world the simulator draws, not what the trader is told.
"""

import hashlib
import inspect
import json
import time
from dataclasses import replace

import numpy as np
import torch

import NN_one_offer_game.torch_train as T
import NN_one_offer_game.rl_diagnostics as D
from Mechanics.fx_mechanics import GameParams, results_path
from MM_Phase_1.train_traders import CARD

# ============================ SETTINGS ======================================
SIDES = ("A", "B")          # both cards, in this order
A0_WINDOW = (1.10, 1.40)    # the opening rate, uniform per block: +/-12%
                            # around the card's 1.25
SD_WINDOW = (0.025, 0.075)  # the volatility, uniform per block: 0.5x to
                            # 1.5x the card's 0.05. Above sigma_c ~ 0.02
                            # (the fee-dodging regime is out); side B's
                            # sigma_dom = a0*sqrt(f)/2 is 0.078 at a0 =
                            # 1.10, so the top corner is inside but close.
                            # (0.05, 0.05) pins it: run 1, see the header
BLOCKS = 5                  # markets per iteration; BATCH / BLOCKS games
                            # share one (a0, sd). Avi's 1,000 in 5 x 200
VAL_BLOCKS = 20             # fixed validation markets, VAL_PATHS / this each
EVAL_BLOCKS = 200           # final window evaluation, EVAL_PATHS / this each
ITERS = T.ITERS             # run length, torch_train's by default. Change
BATCH = T.BATCH             # them HERE, never in the certified file
VAL_PATHS = T.VAL_PATHS
EVAL_PATHS = T.EVAL_PATHS
CHECKS = True               # checks 1-3 must pass before any training
REPORT = True               # rl_diagnostics.report on the card at the end
REPORT_ONLY = False         # skip training: load policy_best.pth of a DONE
                            # side and redo the evaluation and the report
SEED_MARKET = 20_000        # added to torch_train's SEED_PATH for every
                            # (a0, sd) draw, so the market streams never
                            # overlap the shock streams they are paired with
# ============================================================================

torch.set_default_dtype(torch.float64)

LABEL = "generalist"        # this trader's name in the report and figures

# paths this run built, and how many touched zero (dropped, per Avi)
GUARD = {"paths": 0, "dropped": 0}


# ---------------------------------------------------------------------------
# 1. models -- torch_train's, untouched: T.AlphaNet, T.squash, T.build_net
# ---------------------------------------------------------------------------
def window_text():
    # the world this run draws, in one line (a pinned window says so)
    def one(name, lo, hi):
        return (f"{name} pinned at {lo:g}" if lo == hi
                else f"{name} ~ U({lo:g}, {hi:g})")
    return one("a0", *A0_WINDOW) + ", " + one("sd", *SD_WINDOW)


# ---------------------------------------------------------------------------
# 2. the game -- torch_train's rollout on a market of this block's own
# ---------------------------------------------------------------------------
def block_spec(spec, a0, sd):
    # the CARD with this block's market swapped in: same L, T, A, B, side,
    # rounds, K and Bureau fee -- only the two numbers the window varies
    return replace(spec, params=GameParams(a0=float(a0), sd=float(sd),
                                           bdc_fee=spec.params.bdc_fee))


def market_specs(spec, n, generator):
    # n markets from the windows, a0 and sd uniform and independent, one
    # pair per block. A pinned window returns the same number every time.
    u = torch.rand(n, 2, generator=generator)
    a0 = A0_WINDOW[0] + (A0_WINDOW[1] - A0_WINDOW[0]) * u[:, 0]
    sd = SD_WINDOW[0] + (SD_WINDOW[1] - SD_WINDOW[0]) * u[:, 1]
    return [block_spec(spec, a, s) for a, s in zip(a0.tolist(), sd.tolist())]


def levels_of(spec, shocks):
    # the rate path these shocks make under this market, in the SAME ops and
    # order rollout uses: columns 0..R-1 are the hidden rates X_1..X_R,
    # column R the settlement rate. Read by the zero guard and the floor;
    # never by the network. Check 3 pins it to torch_train's convention.
    market = spec.params
    hidden = market.a0 + torch.cumsum(market.sd * shocks[:, :-1], dim=1)
    settlement = hidden[:, -1:] + market.sd * shocks[:, -1:]
    return torch.cat([hidden, settlement], dim=1)


def block_rollout(net, specs, shocks, tau=None):
    # play each block on its own market through torch_train.rollout,
    # unchanged. Returns the kept games' P/L and rate levels, concatenated
    # in batch order. A game whose path touches zero is dropped from both
    # and counted (Avi: get rid of them, do not flip them).
    n_blocks = len(specs)
    assert shocks.shape[0] % n_blocks == 0, "blocks must be equal"
    per_block = shocks.shape[0] // n_blocks
    pls, lvs = [], []
    for k, spec_k in enumerate(specs):
        block = shocks[k * per_block:(k + 1) * per_block]
        levels = levels_of(spec_k, block)
        keep = (levels > 0.0).all(dim=1)
        GUARD["paths"] += per_block
        GUARD["dropped"] += per_block - int(keep.sum())
        pl = T.rollout(net, spec_k, block, tau=tau)
        pls.append(pl[keep])
        lvs.append(levels[keep])
    return torch.cat(pls), torch.cat(lvs)


def block_loss(net, specs, shocks, tau):
    # negation as in torch_train: optimisers minimise, we want P/L maximised.
    # The mean over the kept games of every block = the batch mean.
    pl, _ = block_rollout(net, specs, shocks, tau)
    return -pl.mean()


# ---------------------------------------------------------------------------
# 3. checks -- discharged before training
# ---------------------------------------------------------------------------
def market_corners(spec):
    """The card and the four corners of the window, deduplicated so a pinned
    window does not repeat itself."""
    seen, out = set(), []
    for a0, sd in ([(spec.params.a0, spec.params.sd)]
                   + [(a, s) for a in A0_WINDOW for s in SD_WINDOW]):
        if (a0, sd) not in seen:
            seen.add((a0, sd))
            out.append(block_spec(spec, a0, sd))
    return out


def check_blocks_collapse(spec, n_games=None, seed=3):
    """check 3 -- this file's one new seam.

    (a) With every block pinned to the card, the blocked game must be
    torch_train's whole-batch game, game for game: same rollout, same paths,
    so any gap is a slicing or masking bug. Hard game only: with a ramp,
    rollout measures its width from each block's own std, so smoothed
    losses legitimately differ by that estimate's jitter (see the header).
    (b) levels_of must invert torch_train's shock convention -- the one
    check 1 uses to feed forced rates into rollout -- so the zero guard and
    the always-BdC floor see the rates rollout actually plays."""
    n_games = BATCH if n_games is None else n_games   # BLOCKS divides BATCH
    net = T.build_net(T.SEED_INIT)
    generator = torch.Generator().manual_seed(seed)
    shocks = torch.randn(n_games, spec.rounds + 1, generator=generator)
    pinned = [block_spec(spec, spec.params.a0, spec.params.sd)
              for _ in range(BLOCKS)]
    with torch.no_grad():
        blocked, _ = block_rollout(net, pinned, shocks)
        whole = T.rollout(net, spec, shocks, tau=None)
    gap_pl = float((blocked - whole).abs().max())

    rng = np.random.default_rng(seed)
    a0, sd = spec.params.a0, spec.params.sd
    forced = a0 + sd * np.cumsum(
        rng.standard_normal((n_games, spec.rounds + 1)), axis=1)
    previous = np.concatenate(
        [np.zeros((n_games, 1)), forced[:, :-1] - a0], axis=1)
    inverted = torch.tensor((forced - a0 - previous) / sd)
    gap_lv = float(np.abs(levels_of(spec, inverted).numpy() - forced).max())

    passed = gap_pl < 1e-12 and gap_lv < 1e-9
    print(f"  check 3  blocks pinned to the card vs whole batch, {n_games} "
          f"games (side {spec.side}): max |P/L gap| {gap_pl:.2e}; levels "
          f"vs forced rates: max |gap| {gap_lv:.2e}  "
          f"[{'ok' if passed else 'FAIL'}]")
    assert passed, "the block plumbing does not collapse to torch_train"


def run_checks(spec):
    # checks 1 and 2 are torch_train's, run where THIS file plays: the card
    # and the window's corners. a0 and sd enter rollout only through the
    # path, so passing at the corners certifies the off-card plumbing.
    for corner in market_corners(spec):
        print(f"  market a0 {corner.params.a0:.4f}  sd "
              f"{corner.params.sd:.4f}")
        T.check_torch_game_matches_env(corner)
        T.check_policy_replay_matches_env(corner)
    check_blocks_collapse(spec)


# ---------------------------------------------------------------------------
# 4. training -- torch_train.train with the market drawn per block
# ---------------------------------------------------------------------------
def train(spec, run):
    net = T.build_net(T.SEED_INIT)           # the specialist's starting weights
    optimiser = torch.optim.AdamW(net.parameters(), lr=T.LR)
    # the repository's scheduler: if the loss stops improving, halve the
    # step size rather than keep overshooting
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=200)

    n_anneal = max(int(T.TAU_ANNEAL_FRAC * ITERS), 1)
    decay = (T.TAU_END / T.TAU_START) ** (1.0 / max(n_anneal - 1, 1))

    # the VALIDATION set: one fixed set of WINDOW games -- torch_train's own
    # validation shocks, VAL_BLOCKS markets drawn once -- reused at every
    # check so checkpoints are compared on identical games. The card is
    # scored on the same shocks alongside, for the record and the curve
    # figure only: the checkpoint is chosen on the window, never the card.
    validation_generator = torch.Generator().manual_seed(T.SEED_PATH + 1)
    val_shocks = torch.randn(VAL_PATHS, spec.rounds + 1,
                             generator=validation_generator)
    val_specs = market_specs(spec, VAL_BLOCKS, torch.Generator().manual_seed(
        T.SEED_PATH + SEED_MARKET + 1))
    best_checkpoint = {"window_pl": -1e9, "iteration": -1}
    curve = {"iteration": [], "window_pl": [], "card_pl": []}
    for iteration in range(ITERS):
        # ramp width for this step: anneal, then hold (torch_train's schedule)
        if T.GATE == "hard":
            tau = None
        elif iteration < n_anneal:
            tau = T.TAU_START * decay ** iteration      # shrinking
        else:
            tau = T.TAU_END                             # held

        # STEP 1: draw a FRESH batch: torch_train's own shock stream for this
        # iteration, cut into BLOCKS blocks, and one market per block from
        # this file's stream. New seeds every iteration, so no game is ever
        # seen twice and there is nothing to memorise.
        batch_generator = torch.Generator().manual_seed(
            T.SEED_PATH + 10_000 + iteration)
        shocks = torch.randn(BATCH, spec.rounds + 1, generator=batch_generator)
        specs = market_specs(spec, BLOCKS, torch.Generator().manual_seed(
            T.SEED_PATH + SEED_MARKET + 10_000 + iteration))

        # STEP 2: play every block on its market with the current weights
        # and average the P/L
        loss = block_loss(net, specs, shocks, tau)
        if not torch.isfinite(loss):
            print(f"  [guard] loss non-finite at iter {iteration}. "
                  f"Skipping step.")
            optimiser.zero_grad()
            continue

        # STEP 3: backprop
        optimiser.zero_grad()
        loss.backward()

        # STEP 4: cap the step's size, then take it
        torch.nn.utils.clip_grad_norm_(net.parameters(), T.CLIP_NORM)
        optimiser.step()
        lr_scheduler.step(loss.item())

        # STEP 5: periodically score the policy on the TRUE game (tau=None)
        # on the fixed validation games, and keep the best version so far
        if iteration % T.VAL_EVERY == 0 or iteration == ITERS - 1:
            with torch.no_grad():
                window_pl = float(
                    block_rollout(net, val_specs, val_shocks)[0].mean())
                card_pl = float(
                    T.rollout(net, spec, val_shocks, tau=None).mean())
            curve["iteration"].append(iteration)
            curve["window_pl"].append(window_pl)
            curve["card_pl"].append(card_pl)
            tau_label = "  hard " if tau is None else f"tau {tau:8.5f}"

            # 'batch P/L' is a 1,000-path estimate of the SMOOTHED game and
            # is noisy by design; the hard numbers are 20,000 paths of the
            # real one -- the window is what the checkpoint is chosen on
            print(f"  iter {iteration:>5}  {tau_label}  "
                  f"batch P/L {-loss.item():+.5f}  "
                  f"hard P/L window {window_pl:+.5f}  card {card_pl:+.5f}")
            if window_pl > best_checkpoint["window_pl"]:
                best_checkpoint.update(window_pl=window_pl,
                                       iteration=iteration)
                torch.save(net.state_dict(), run / "policy_best.pth")
    torch.save(net.state_dict(), run / "policy_final.pth")

    # early stopping: evaluate the checkpoint that validated best on the
    # hard WINDOW game -- the late anneal is the fragile stage (torch_train)
    print(f"  best hard window validation "
          f"{best_checkpoint['window_pl'] * 100:+.4f}% at iter "
          f"{best_checkpoint['iteration']}; evaluating policy_best.pth "
          f"(final kept on disk)")
    net.load_state_dict(torch.load(run / "policy_best.pth"))
    return net, curve


def _fingerprint(spec):
    """Everything that determines what a trained generalist IS, after
    train_traders._trader_fingerprint: the card, the windows and blocks,
    torch_train's recipe as it is NOW, and the source of the functions that
    play the game. A DONE marker is honoured only for the configuration
    that produced it."""
    payload = json.dumps({
        "card": [spec.L, spec.T, spec.A, spec.B, spec.side, spec.rounds,
                 spec.K],
        "market": [spec.params.a0, spec.params.sd, spec.params.bdc_fee],
        "windows": [list(A0_WINDOW), list(SD_WINDOW)],
        "blocks": [BLOCKS, VAL_BLOCKS],
        "iters": ITERS, "batch": BATCH, "val_paths": VAL_PATHS,
        "recipe": [T.LR, T.CLIP_NORM, T.GATE, T.TAU_START, T.TAU_END,
                   T.TAU_ANNEAL_FRAC, T.VAL_EVERY, T.LAGS],
        "seeds": [T.SEED_INIT, T.SEED_PATH, SEED_MARKET],
        "src": hashlib.sha256("".join(
            inspect.getsource(f) for f in
            (T.AlphaNet, T.squash, T.features, T.rollout, block_spec,
             market_specs, levels_of, block_rollout, train)
        ).encode()).hexdigest()[:16],
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 5. evaluation and main
# ---------------------------------------------------------------------------
def _pm(x):
    # mean and two standard errors, as rl_diagnostics prints them
    x = np.asarray(x, float)
    return float(x.mean()), float(2.0 * x.std(ddof=1) / np.sqrt(len(x)))


def final_evaluation(net, spec):
    """The two headline numbers on the torch engine, hard game, fresh paths
    never trained or validated on: the wider world (EVAL_PATHS window games
    in EVAL_BLOCKS markets) and the card, on the SAME shocks. Both are
    floor-anchored by the paired always-BdC counterfactual (D.dump_all_pl:
    closed form per path, card inputs only, so it needs no market). Printed,
    and returned as lines for the summary."""
    R = spec.rounds
    eval_generator = torch.Generator().manual_seed(T.SEED_PATH + 2)
    shocks = torch.randn(EVAL_PATHS, R + 1, generator=eval_generator)
    specs = market_specs(spec, EVAL_BLOCKS, torch.Generator().manual_seed(
        T.SEED_PATH + SEED_MARKET + 2))
    with torch.no_grad():
        pl_window, levels_window = block_rollout(net, specs, shocks)
        pl_card = T.rollout(net, spec, shocks, tau=None)
    levels_card = levels_of(spec, shocks)

    def floor(levels):
        return D.dump_all_pl(spec, levels[:, 0].numpy(), levels[:, R].numpy())

    rows = (("window", pl_window.numpy(), floor(levels_window)),
            ("card", pl_card.numpy(), floor(levels_card)))
    lines = [f"\nTHE WIDER WORLD  (torch engine, hard game, {EVAL_PATHS:,} "
             f"fresh paths in {EVAL_BLOCKS} markets,",
             f"{window_text()}; the card played on the SAME shocks)",
             f"  {'':<26} {'mean P/L':>18}   {'over always-BdC (paired)':>26}"]
    for name, pl, fl in rows:
        m, pm = _pm(pl)
        e, pe = _pm(pl - fl)
        lines.append(f"  {LABEL + ', ' + name:<26} {m * 100:+9.4f}% +/- "
                     f"{pm * 100:.4f}%   {e * 100:+9.4f}% +/- {pe * 100:.4f}%")
    lines.append(f"  paths touching zero: {GUARD['dropped']:,} of "
                 f"{GUARD['paths']:,} built this run (dropped, per Avi)")
    print("\n".join(lines))
    return lines


def main():
    for side in SIDES:
        spec = CARD[side]                    # the card, stated in Phase 1
        run = results_path(f"phase2_1/trader_{side}_A0_{A0_WINDOW}_SD_{SD_WINDOW}")
        run.mkdir(parents=True, exist_ok=True)
        done_marker = run / "DONE"
        fingerprint = _fingerprint(spec)
        fresh = (done_marker.exists()
                 and done_marker.read_text().strip() == fingerprint)
        if REPORT_ONLY:
            if not fresh:
                print(f"side {side}: REPORT_ONLY but no trader matches the "
                      f"current card, windows and recipe; train first",
                      flush=True)
                continue
        elif done_marker.exists() and not fresh:
            print(f"side {side}: STALE trader -- the DONE fingerprint does "
                  f"not match the current card, windows or recipe. Discarding "
                  f"and retraining; reusing it would score a trader built for "
                  f"a DIFFERENT world.", flush=True)
            for stale in ("policy_best.pth", "policy_final.pth", "curve.npz"):
                (run / stale).unlink(missing_ok=True)
            done_marker.unlink()
        elif fresh:
            print(f"side {side}: DONE fingerprint matches, skipping "
                  f"(REPORT_ONLY = True to redo the report)", flush=True)
            continue

        print(f"\n################ PHASE 2.1  side {side}: card L={spec.L:,.0f} "
              f"T={spec.T:,.0f} A={spec.A} B={spec.B} rounds={spec.rounds} "
              f"K={spec.K} ################", flush=True)
        print(f"  the world: {window_text()}"
              + ("  (run 1: Avi's initial-random, sigma fixed)"
                 if SD_WINDOW[0] == SD_WINDOW[1] else ""))
        print(f"  budget: {ITERS:,} iters x {BATCH:,} paths in {BLOCKS} "
              f"blocks of {BATCH // BLOCKS} = {ITERS * BATCH:,} games over "
              f"{ITERS * BLOCKS:,} markets")
        print("  method: torch_train's rollout and network, unchanged -- "
              "direct policy optimisation through the differentiable "
              "simulator, the market drawn per block")
        assert BATCH % BLOCKS == 0 and VAL_PATHS % VAL_BLOCKS == 0 \
            and EVAL_PATHS % EVAL_BLOCKS == 0, "blocks must divide the paths"
        assert A0_WINDOW[0] <= A0_WINDOW[1] and SD_WINDOW[0] <= SD_WINDOW[1] \
            and A0_WINDOW[0] > 0 and SD_WINDOW[0] > 0, "windows must be ordered and positive"

        if CHECKS:
            run_checks(spec)
        GUARD.update(paths=0, dropped=0)     # the checks' games are not the run's
        start_time = time.time()

        if REPORT_ONLY:
            net = T.build_net(T.SEED_INIT)
            net.load_state_dict(torch.load(run / "policy_best.pth"))
            net.eval()
            saved = np.load(run / "curve.npz")
            curve = {key: saved[key].tolist() for key in saved.files}
            print(f"  REPORT_ONLY: loaded policy_best.pth "
                  f"(fingerprint {fingerprint})")
        else:
            net, curve = train(spec, run)
            np.savez(run / "curve.npz",
                     iteration=np.array(curve["iteration"]),
                     window_pl=np.array(curve["window_pl"]),
                     card_pl=np.array(curve["card_pl"]))
            done_marker.write_text(fingerprint)
            print(f"  trained in {(time.time() - start_time) / 60:.2f} min; "
                  f"DONE {fingerprint}")

        extra = final_evaluation(net, spec)

        # the card, through the certified env: rl_diagnostics solves the DP
        # fresh, replays the trader and the DP on the same paths, writes the
        # five standard outputs. The curve drawn is the CARD's validation
        # series -- the one the figure's reference lines belong to -- with
        # the wider-world numbers appended to the summary.
        if REPORT:
            D.report(spec, T.torch_policy(net, spec), run, label=LABEL,
                     curve=(curve["iteration"], curve["card_pl"],
                            "iteration"),
                     extra=extra)
        print(f"artefacts in {run}/  "
              f"({(time.time() - start_time) / 60:.2f} min)", flush=True)
    print("\nPHASE 2.1 DONE")


if __name__ == "__main__":
    main()