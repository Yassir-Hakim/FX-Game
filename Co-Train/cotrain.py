import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "NN_one_offer_game"      # as mm_train.py; edit this ONE name
_PHASE1 = _ROOT / "MM_Phase_1"           # if either folder is ever renamed
assert (_CORE / "rl_env.py").exists(), f"core folder not found: {_CORE}"
assert (_PHASE1 / "mm_train.py").exists(), f"Phase 1 folder not found: {_PHASE1}"
sys.path.insert(0, str(_CORE))           # the certified machinery layer
sys.path.insert(0, str(_ROOT))           # Mechanics/, DP_Models/
sys.path.insert(0, str(_PHASE1))         # mm_train and train_traders import
                                         # each other by bare name
"""
  cotrain.py -- the trader and the market maker learn AGAINST EACH OTHER.
  Both start from fresh weights. Blocks alternate: one agent updates for
  N_INNER iterations while the other is frozen, then they swap, N_OUTER
  times. One game (mm_train.rollout, unchanged) and two objectives from
  the same play: the trader's P/L and the MM's.
    Phase 1 froze a trader that already expected the naive rule and found
  an MM that agreed with it. That is AN equilibrium. This file asks
  whether a free pair finds THE SAME one on its own, and tests it by
  cross-play: on the same paths, does either Phase 1 agent beat its
  co-trained counterpart against the co-trained partner?

WHAT CHANGES AGAINST PHASE 1, AND WHY (each a design note in the plan)
  - nothing is pre-trained: pre-training the trader would precondition it
    to the answer (Avi, meeting 4)
  - the MM moves first by default: build_mm_net starts the level 10%
    conservative (rejects everything); a trader learning against that
    routes everything to the Bureau and the MM then has no flow to learn
    from. MM_LEVEL_INIT = "random" restores Avi's literal recipe
  - the trader keeps torch_train's plateau LR decay (re-derived: see
    TRADER_PLATEAU)
  - no "best" checkpoint: there is no single objective; the PAIR is saved
  - the four reference rules are recomputed at every validation: they are
    P/L against the CURRENT trader and move as it learns
  - a0 is fixed (mm_train.rollout reads it from the card); the a0 window
    is the next variant once this reproduces

MEASUREMENT (project standard, not learner features)
  Phase 1's three checks run on the fresh trader (they hold for any trader
  net), a fourth checks the freeze seam, and a fifth ties the trader's
  floor to v2's analytic always-BdC. Every reported number is a hard-game
  replay, paired on identical paths -- INCLUDING the trader's floor: the
  always-BdC P/L is evaluated on the same paths as the trader, because the
  raw P/L carries the rate path (~ +/-0.1% on 20,000 paths) and an unpaired
  floor line cannot tell a trader that trades from one that dumps. The
  learner is told nothing by any of this.

DISTRIBUTION KNOWLEDGE: none, as mm_train and torch_train.
"""

import time

import numpy as np
import torch

import MM_Phase_1.mm_train as M                          # Phase 1: game, MM net, rules,
                                              # checks, paired, measurement
import NN_one_offer_game.torch_train as T     # the trader's net; the SAME
                                              # module object mm_train uses
from NN_one_offer_game.rl_diagnostics import dump_all_pl   # always-BdC per
                                              # path, closed form, certified
                                              # to 1e-9 against the env
from DP_Models.v2_multiple_rounds import bdc_baseline      # the analytic
                                              # floor: check 5 and the summary
from MM_Phase_1.train_traders import CARD
from Mechanics.fx_mechanics import results_path

# ============================ SETTINGS ======================================
SIDES = ("A", "B")
MM_FEE = 0.01           # passed to EVERY rollout; never read from M.MM_FEE.
                        # The Phase 1 MM for the cross-play must have been
                        # trained at this fee (its run folder carries it)
N_OUTER = 20            # one outer step = one block for each agent
N_INNER = 100           # iterations per block. 20 x 100 = Phase 1's 2000 per
                        # agent: a starting budget, not a derived one (plan,
                        # design note 4). Test 50 and 200
FIRST = "mm"            # who updates first. "mm": the MM learns against
                        # random quotes that straddle its level. "trader"
                        # (Avi's literal recipe) needs MM_LEVEL_INIT =
                        # "random" (design note 3)
MM_LEVEL_INIT = "conservative"  # "conservative": mm_train's build_mm_net,
                        # level started LEVEL_INIT_OFFSET off on the
                        # rejecting side. "random": the head as initialised,
                        # nothing set -- the random MM of Avi's recipe
BATCH = 1000            # games per iteration, as both arms
LR_TRADER = 1e-3        # torch_train's value; constant here (see below)
LR_MM = 1e-4            # mm_train's
TRADER_PLATEAU = True   # torch_train's ReduceLROnPlateau, kept. The plan
                        # argued to drop it (non-stationary loss); measured,
                        # that argument LOSES: the MM settles within ~2 outer
                        # steps, and without the decay the constant-LR trader
                        # random-walks off the shallow interior optimum into
                        # the absorbing no-trade region (seen at fee 0).
                        # Re-derived, not inherited. Off = the old behaviour
TAU_TRADER = (1.0, 0.02)   # (start, end) ramp, torch_train's
TAU_MM = (0.2, 0.02)       # mm_train's. A block runs on the LEARNER's ramp,
                           # in the observed rate-move units of the batch
TAU_ANNEAL_FRAC = 0.35     # of each agent's OWN N_OUTER * N_INNER steps,
                           # then hold, as both arms
CLIP_NORM = 1.0
VAL_PATHS = 20_000      # hard-game validation after EVERY block
EVAL_PATHS = 200_000    # final paired evaluation and the cross-play
RECORD_PATHS = 20_000   # fresh paths replayed WITH a record, for the
                        # diagnostics (levels, offers, boundary plots)
CONVERGED_BLOCKS = 4    # stop early once, over this many consecutive outer
                        # steps, every consecutive change of BOTH hard P/Ls
                        # is within max(its paired 2 SE, the tolerance below).
                        # Reported either way
FLAT_TOL_MM = 2e-6      # 0.02 bp of capital: a paired test on 20,000 shared
FLAT_TOL_TRADER = 5e-6  # paths resolves changes far below anything that
                        # matters, so flatness needs an economic floor too.
                        # 5e-6 of L is 0.0005%
EVAL_ONLY = False       # True: skip training, load pair_final.pth (or
                        # pair_last.pth) and cotrain_curve.npz from the run
                        # folder and evaluate. For re-running the evaluation
                        # of a finished run
CHECKS = True           # checks 1-4 must pass before any training
CROSS_PLAY = True       # evaluate against Phase 1's frozen trader and
                        # learned MM. Needs train_traders.py and mm_train.py
                        # to have run for the side at this fee
SEED_INIT_TRADER = 42   # fresh trader, T.build_net
SEED_INIT_MM = 42       # fresh MM, M.build_mm_net
SEED_PATH = 9012        # this file's own path noise; not 1234 (torch_train)
                        # nor 5678 (mm_train)
SET_PATH = [1.30, 1.36, 1.37, 1.41, 1.33]   # the demo game's rates, rounds
                        # 1-4 then settlement, for the set-path figure
# ============================================================================

torch.set_default_dtype(torch.float64)
assert N_OUTER * N_INNER <= 10_000, "batch seeds would overlap; see train_block"
assert MM_LEVEL_INIT in ("conservative", "random")
assert FIRST in ("mm", "trader")


# ---------------------------------------------------------------------------
# 1. models -- both fresh; nothing loaded, nothing pre-trained
# ---------------------------------------------------------------------------
def build_mm_random(seed):
    # the head exactly as torch initialises it: the level lands wherever the
    # random weights put it. Avi's "randomise the MM's weights"
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return T.AlphaNet(n_in=M.N_MM_FEAT, n_out=2)


def build_pair(side):
    trader_net = T.build_net(SEED_INIT_TRADER)
    mm_net = (M.build_mm_net(SEED_INIT_MM, side)
              if MM_LEVEL_INIT == "conservative"
              else build_mm_random(SEED_INIT_MM))
    return trader_net, mm_net


def set_learning(net, learning):
    # the frozen agent still runs forward, so the learner's gradient passes
    # through its decisions; only its weights stop moving
    for parameter in net.parameters():
        parameter.requires_grad_(learning)
    if not learning:
        net.zero_grad(set_to_none=True)    # no stale .grad from its last block
    net.train() if learning else net.eval()


def tau_for(agent, step):
    # each agent anneals over its OWN cumulative steps, then holds
    start, end = TAU_TRADER if agent == "trader" else TAU_MM
    n_anneal = max(int(TAU_ANNEAL_FRAC * N_OUTER * N_INNER), 1)
    decay = (end / start) ** (1.0 / max(n_anneal - 1, 1))
    return start * decay ** step if step < n_anneal else end


def order_of_play():
    return ("mm", "trader") if FIRST == "mm" else ("trader", "mm")


# ---------------------------------------------------------------------------
# 2. the game -- mm_train.rollout as it is; one play, two objectives
# ---------------------------------------------------------------------------
def block_loss(agent, mm_net, trader_net, spec, rate_moves_sd, tau):
    #negation is because optimisers MINIMISE and we want to MAXIMISE the payoff
    mm_pl, trader_pl, _ = M.rollout(mm_net, trader_net, spec, rate_moves_sd,
                                    tau=tau, fee=MM_FEE)
    return -(trader_pl if agent == "trader" else mm_pl).mean()


def always_bdc_pl(spec, rate_moves_sd):
    """The trader's outside option on THESE paths: nothing to the MM, the
    whole book at the round-1 Bureau rate, the proceeds settle. Closed form
    per path (rl_diagnostics.dump_all_pl), on the rates rollout builds from
    the same z-draws, so it pairs with every P/L measured on rate_moves_sd.
    Returns a tensor, as rollout does, so M.paired applies."""
    market = spec.params
    z = rate_moves_sd.detach().numpy()
    hidden_rates = market.a0 + np.cumsum(market.sd * z[:, :spec.rounds], axis=1)
    settlement_rate = hidden_rates[:, -1] + market.sd * z[:, spec.rounds]
    return torch.as_tensor(dump_all_pl(spec, hidden_rates[:, 0], settlement_rate))


# ---------------------------------------------------------------------------
# 3. checks -- Phase 1's three on the fresh trader, plus the freeze seam
# ---------------------------------------------------------------------------
def _snapshot(net):
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


def _unchanged(before, net):
    return all(torch.equal(before[key], value)
               for key, value in net.state_dict().items())


def check_freeze_seam(spec, side):
    """One step for either agent must leave the other bit-identical AND must
    move the learner: the seam between the two optimisers is the one new
    piece of plumbing in this file."""
    trader_net, mm_net = build_pair(side)
    gen = torch.Generator().manual_seed(SEED_PATH + 4)
    batch = torch.randn(BATCH, spec.rounds + 1, generator=gen)
    for agent in ("mm", "trader"):
        learner, other = ((mm_net, trader_net) if agent == "mm"
                          else (trader_net, mm_net))
        optimiser = torch.optim.AdamW(learner.parameters(), lr=1e-3)
        set_learning(learner, True)
        set_learning(other, False)
        learner_before, other_before = _snapshot(learner), _snapshot(other)
        loss = block_loss(agent, mm_net, trader_net, spec, batch,
                          tau_for(agent, 0))
        optimiser.zero_grad()
        loss.backward()
        assert all(p.grad is None for p in other.parameters()), (
            f"check 4: the frozen {'trader' if agent == 'mm' else 'MM'} "
            f"received gradient")
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in learner.parameters()), (
            f"check 4: the learning {agent} received NO gradient -- its "
            f"objective is flat in its own weights on this batch")
        optimiser.step()
        assert _unchanged(other_before, other), (
            f"check 4: the frozen {'trader' if agent == 'mm' else 'MM'} moved")
        assert not _unchanged(learner_before, learner), (
            f"check 4: the learning {agent} did not move")
    print(f"  check 4  freeze seam (side {side}): each agent's step moves "
          f"itself only, and the frozen partner passes gradient through  [ok]",
          flush=True)


def check_bdc_floor(spec):
    """Check 5: the per-path always-BdC that floors every trader number in
    this file must average to v2's analytic floor -- closed form against
    Monte Carlo, the project's verification standard. On the evaluation
    paths, so the number here is the summary's."""
    _, floor = bdc_baseline(spec)
    gen = torch.Generator().manual_seed(SEED_PATH + 2)
    pl = always_bdc_pl(spec, torch.randn(EVAL_PATHS, spec.rounds + 1,
                                         generator=gen))
    mean, two_se = float(pl.mean()), 2 * float(pl.std()) / np.sqrt(EVAL_PATHS)
    assert abs(mean - floor) <= two_se, (
        f"check 5: always-BdC per path {mean * 100:+.4f}% +/- {two_se * 100:.4f}% "
        f"disagrees with the analytic floor {floor * 100:+.4f}% -- the paths, "
        f"the card, or dump_all_pl's settlement differ from rollout's")
    print(f"  check 5  always-BdC floor (side {spec.side}): analytic "
          f"{floor * 100:+.4f}%, per path on {EVAL_PATHS:,} eval paths "
          f"{mean * 100:+.4f}% +/- {two_se * 100:.4f}%  [ok]", flush=True)


# ---------------------------------------------------------------------------
# 4. training -- alternating blocks
# ---------------------------------------------------------------------------
def train_block(agent, mm_net, trader_net, optimiser, scheduler, spec, step):
    learner, other = ((trader_net, mm_net) if agent == "trader"
                      else (mm_net, trader_net))
    set_learning(learner, True)
    set_learning(other, False)
    last_loss = float("nan")
    for _ in range(N_INNER):
        tau = tau_for(agent, step)
        # STEP 1: draw a FRESH batch. The seed is keyed on the agent as well
        # as the step, so the two never train on the same paths
        offset = 10_000 if agent == "trader" else 20_000
        gen = torch.Generator().manual_seed(SEED_PATH + offset + step)
        batch_rate_moves_sd = torch.randn(BATCH, spec.rounds + 1, generator=gen)
        step += 1

        # STEP 2: play them all with the current weights, average the
        # LEARNER's P/L
        loss = block_loss(agent, mm_net, trader_net, spec, batch_rate_moves_sd,
                          tau)
        if not torch.isfinite(loss):
            print(f"  [guard] {agent} loss non-finite at its step {step - 1}. "
                  f"Skipping step.", flush=True)
            optimiser.zero_grad()
            continue

        # STEP 3: backprop -- through the frozen partner's decisions
        optimiser.zero_grad()
        loss.backward()

        # STEP 4: cap the step's size, then take it
        torch.nn.utils.clip_grad_norm_(learner.parameters(), CLIP_NORM)
        optimiser.step()
        if scheduler is not None:
            scheduler.step(loss.item())
        last_loss = -loss.item()
    return step, tau, last_loss


def validate(mm_net, trader_net, spec, val_rate_moves_sd, bdc_pl):
    """The HARD game on the fixed validation paths. The rule references are
    recomputed every time: they are P/L against the CURRENT trader, so
    unlike Phase 1's fixed lines they move as the trader learns. The
    benchmark rule is the best of them on these paths, as mm_train.
    bdc_pl: always-BdC on these same paths (fixed, computed once), so the
    trader's edge over its outside option is paired."""
    set_learning(mm_net, False)
    set_learning(trader_net, False)
    rules = M.reference_rules(MM_FEE)
    with torch.no_grad():
        mm_pl, trader_pl, log = M.rollout(mm_net, trader_net, spec,
                                          val_rate_moves_sd, tau=None,
                                          fee=MM_FEE, record=True)
        rule_pl = {rule.name: M.rollout(None, trader_net, spec,
                                        val_rate_moves_sd, rule=rule,
                                        fee=MM_FEE)[0]
                   for rule in rules}
    benchmark_rule = max(rules, key=lambda rule: float(rule_pl[rule.name].mean()))
    agree, filled, filled_benchmark, benchmark_name = M.volume_agreement(
        log, spec.side, benchmark_rule, fee=MM_FEE)
    # the level as SET (a policy, fixed before the offer is read), per round
    level = np.median(log["break_even"] / log["hidden_rate"], axis=0)
    # the trader's opening decision: the round-1 quote in $/GBP -- the
    # game's rate on BOTH sides (rollout banks filled / price on side B) --
    # and the fraction of its book it puts up
    offer_rate = float(np.median(log["quoted_price"][:, 0]))
    fraction = float(np.median(log["offered_amount"][:, 0])) / spec.L
    # the trader against its outside option on THESE paths. This bar is the
    # readable one: the raw trader_pl's carries the rate path
    edge, edge_2se = M.paired(trader_pl, bdc_pl)
    return dict(mm_pl=mm_pl, trader_pl=trader_pl, rule_pl=rule_pl,
                bdc_pl=bdc_pl, edge=edge, edge_2se=edge_2se,
                level=level, offer_rate=offer_rate, fraction=fraction,
                agree=agree, filled=filled, filled_benchmark=filled_benchmark,
                benchmark=benchmark_name, log=log)


def flat_run(recent):
    """recent: the (mm_pl, trader_pl) validation vectors after the last few
    outer steps. Flat when every consecutive PAIRED change of both P/Ls is
    within max(2 SE, tolerance): neither agent is still changing the other's
    answer by an amount that matters."""
    if len(recent) < CONVERGED_BLOCKS:
        return False
    for (mm_a, tr_a), (mm_b, tr_b) in zip(recent[:-1], recent[1:]):
        for later, earlier, tolerance in ((mm_b, mm_a, FLAT_TOL_MM),
                                          (tr_b, tr_a, FLAT_TOL_TRADER)):
            change, two_se = M.paired(later, earlier)
            if abs(change) > max(two_se, tolerance):
                return False
    return True


def cotrain(spec, run):
    side = spec.side
    trader_net, mm_net = build_pair(side)
    opt_trader = torch.optim.AdamW(trader_net.parameters(), lr=LR_TRADER)
    sched_trader = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_trader, factor=0.5, patience=200) if TRADER_PLATEAU else None)
    opt_mm = torch.optim.AdamW(mm_net.parameters(), lr=LR_MM)

    # the VALIDATION set: one fixed batch of paths, reused after every block
    # so blocks are compared on identical games. Its own seed
    validation_generator = torch.Generator().manual_seed(SEED_PATH + 1)
    val_rate_moves_sd = torch.randn(VAL_PATHS, spec.rounds + 1,
                                    generator=validation_generator)
    val_bdc_pl = always_bdc_pl(spec, val_rate_moves_sd)   # the floor on them
    rule_names = [rule.name for rule in M.reference_rules(MM_FEE)]

    # block 0: the pair as initialised, so the curves start where the
    # learning starts
    v = validate(mm_net, trader_net, spec, val_rate_moves_sd, val_bdc_pl)
    history = [_row(-1, "init", {"trader": 0, "mm": 0}, v, rule_names)]
    _print_row(history[-1], spec)

    step = {"trader": 0, "mm": 0}
    recent, stopped_at = [], None
    for outer in range(N_OUTER):
        for agent in order_of_play():
            opt, sched = ((opt_trader, sched_trader) if agent == "trader"
                          else (opt_mm, None))
            step[agent], tau, batch_pl = train_block(
                agent, mm_net, trader_net, opt, sched, spec, step[agent])
            v = validate(mm_net, trader_net, spec, val_rate_moves_sd,
                         val_bdc_pl)
            history.append(_row(outer, agent, step, v, rule_names,
                                tau=tau, batch_pl=batch_pl))
            _print_row(history[-1], spec)
        # NO 'best' checkpoint: there is no single objective. The PAIR is
        # saved together -- a trader means nothing without the MM it was
        # measured against
        torch.save({"trader": trader_net.state_dict(),
                    "mm": mm_net.state_dict(), "outer": outer,
                    "steps": dict(step)}, run / "pair_last.pth")
        recent = (recent + [(v["mm_pl"], v["trader_pl"])])[-CONVERGED_BLOCKS:]
        if flat_run(recent):
            stopped_at = outer
            print(f"  flat over the last {CONVERGED_BLOCKS} outer steps "
                  f"(every paired change of both P/Ls within max(2 SE, "
                  f"tolerance)): stopping at outer step {outer}", flush=True)
            break
    if stopped_at is None:
        print(f"  ran all {N_OUTER} outer steps; last {CONVERGED_BLOCKS} "
              f"{'flat' if flat_run(recent) else 'NOT flat'} by the paired "
              f"max(2 SE, tolerance) test", flush=True)
    curve = {key: np.array([row[key] for row in history])
             for key in history[0]}
    curve["stopped_at"] = -1 if stopped_at is None else stopped_at
    np.savez(run / "cotrain_curve.npz", **curve)
    return trader_net, mm_net, curve


def _row(outer, moved, step, v, rule_names, tau=float("nan"),
         batch_pl=float("nan")):
    #one line of the curve: who moved, then everything validate measured
    mm_pl, trader_pl = v["mm_pl"], v["trader_pl"]
    return dict(
        outer=outer, moved=moved, step_trader=step["trader"],
        step_mm=step["mm"], tau=tau, batch_pl=batch_pl,
        mm_pl=float(mm_pl.mean()),
        mm_2se=2 * float(mm_pl.std()) / np.sqrt(len(mm_pl)),
        trader_pl=float(trader_pl.mean()),
        trader_2se=2 * float(trader_pl.std()) / np.sqrt(len(trader_pl)),
        bdc_pl=float(v["bdc_pl"].mean()),     # constant: the paths are fixed
        edge=float(v["edge"]), edge_2se=float(v["edge_2se"]),
        rule_pl=np.array([float(v["rule_pl"][name].mean())
                          for name in rule_names]),
        rule_names=np.array(rule_names),
        level=v["level"], offer_rate=v["offer_rate"], fraction=v["fraction"],
        agree=v["agree"], filled=v["filled"],
        filled_benchmark=v["filled_benchmark"], benchmark=v["benchmark"])


def _print_row(row, spec):
    best = row["rule_pl"].max()
    tau_label = ("  init " if row["moved"] == "init"
                 else f"tau {row['tau']:7.4f}")
    print(f"  outer {row['outer']:>3} after {row['moved']:<6} {tau_label}  "
          f"MM {row['mm_pl'] * 1e4:+8.2f} bp (best rule {best * 1e4:+7.2f})  "
          f"trader {row['trader_pl'] * 100:+8.4f}% "
          f"(vs BdC {row['edge'] * 100:+.4f})  "
          f"r1 level/X {row['level'][0]:.4f}  r1 offer {row['offer_rate']:.4f} $/GBP  "
          f"put up {row['fraction'] * 100:5.1f}%  "
          f"filled {row['filled'] * 100:5.1f}%/{row['filled_benchmark'] * 100:5.1f}%  "
          f"agree({row['benchmark']}) {row['agree'] * 100:5.1f}%", flush=True)


# ---------------------------------------------------------------------------
# 5. main -- every number a paired hard-game replay; the cross-play against
#    Phase 1's pair is the equilibrium test
# ---------------------------------------------------------------------------
def load_phase1(side):
    """Phase 1's frozen trader (fingerprinted, refused if stale) and its
    learned MM at THIS file's fee. Returns None if either is missing, so a
    side can be co-trained before its Phase 1 run exists."""
    mm_file = results_path(f"mm_phase1/mm_{side}_fee{MM_FEE:g}") / "mm_best.pth"
    trader_marker = results_path(f"mm_phase1/trader_{side}") / "DONE"
    if not (mm_file.exists() and trader_marker.exists()):
        print(f"  [cross-play skipped] Phase 1 artefacts missing for side "
              f"{side} at fee {MM_FEE:g}: need {trader_marker} and {mm_file}",
              flush=True)
        return None
    _, trader_net, fingerprint = M.load_trader(side)
    mm_net = M.build_mm_net(M.SEED_INIT, side)
    mm_net.load_state_dict(torch.load(mm_file))
    set_learning(mm_net, False)
    return trader_net, mm_net, fingerprint


def run_side(side):
    # ---- 1. WHICH GAME ----------------------------------------------------
    spec = CARD[side]
    run = results_path(f"cotrain/{side}_fee{MM_FEE:g}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"\n################ side {side}: card L={spec.L:,.0f} T={spec.T:,.0f} "
          f"A={spec.A} B={spec.B} rounds={spec.rounds} K={spec.K}  MM fee "
          f"{MM_FEE} ################", flush=True)
    print(f"  co-training: {N_OUTER} outer steps x {N_INNER} iterations x "
          f"{BATCH:,} games per agent, {FIRST} first, MM level init "
          f"{MM_LEVEL_INIT}; trader LR {LR_TRADER}"
          f"{' + plateau decay' if TRADER_PLATEAU else ' constant'}, MM LR "
          f"{LR_MM}; ramps trader {TAU_TRADER[0]} -> {TAU_TRADER[1]}, MM "
          f"{TAU_MM[0]} -> {TAU_MM[1]} over each agent's first "
          f"{TAU_ANNEAL_FRAC:.0%}", flush=True)
    print("  method: direct policy optimisation through a differentiable "
          "simulator, each agent against the other's CURRENT policy -- NOT "
          "model-free RL, nothing pre-trained", flush=True)

    # ---- 2. THE CHECKS ----------------------------------------------------
    if CHECKS:
        fresh_trader, _ = build_pair(side)
        M.check_trader_game_matches_torch_train(spec, fresh_trader)
        M.check_books_conserve(spec, fresh_trader)
        M.check_rule_pl_closed_form(spec, fresh_trader)
        check_freeze_seam(spec, side)
        check_bdc_floor(spec)
    start_time = time.time()

    # ---- 3. TRAINING ------------------------------------------------------
    if EVAL_ONLY:
        trader_net, mm_net, curve = load_pair(side, run)
    else:
        trader_net, mm_net, curve = cotrain(spec, run)
    set_learning(trader_net, False)
    set_learning(mm_net, False)
    return evaluate_side(spec, run, trader_net, mm_net, curve, start_time)


def load_pair(side, run):
    #a finished run's pair and curve, for EVAL_ONLY
    file = run / "pair_final.pth"
    if not file.exists():
        file = run / "pair_last.pth"
    assert file.exists(), f"EVAL_ONLY: no saved pair in {run}"
    saved = torch.load(file)
    trader_net, mm_net = build_pair(side)
    trader_net.load_state_dict(saved["trader"])
    mm_net.load_state_dict(saved["mm"])
    curve = np.load(run / "cotrain_curve.npz", allow_pickle=False)
    curve = {key: curve[key] for key in curve.files}
    print(f"  EVAL_ONLY: pair loaded from {file.name}, curve of "
          f"{len(curve['mm_pl'])} blocks", flush=True)
    return trader_net, mm_net, curve


def evaluate_side(spec, run, trader_net, mm_net, curve, start_time):
    side = spec.side
    # ---- 4. EVALUATION ----------------------------------------------------
    # a large fresh sample, never seen in training or validation, on the
    # TRUE game: the pair, every rule against its trader, and the cross-play
    # with Phase 1's pair, all on the SAME paths
    rules = M.reference_rules(MM_FEE)
    eval_generator = torch.Generator().manual_seed(SEED_PATH + 2)
    eval_rate_moves_sd = torch.randn(EVAL_PATHS, spec.rounds + 1,
                                     generator=eval_generator)
    eval_bdc_pl = always_bdc_pl(spec, eval_rate_moves_sd)   # the floor, paired
    bdc_floor = float(bdc_baseline(spec)[1])                # and analytic
    phase1 = load_phase1(side) if CROSS_PLAY else None
    plays = {"co MM v co trader": (mm_net, trader_net)}
    if phase1 is not None:
        p1_trader, p1_mm, p1_fingerprint = phase1
        plays.update({"P1 MM v P1 trader": (p1_mm, p1_trader),
                      "P1 MM v co trader": (p1_mm, trader_net),
                      "co MM v P1 trader": (mm_net, p1_trader)})
    with torch.no_grad():
        eval_pl = {name: M.rollout(mm, trader, spec, eval_rate_moves_sd,
                                   tau=None, fee=MM_FEE)[:2]
                   for name, (mm, trader) in plays.items()}
        eval_rule_pl = {rule.name: M.rollout(None, trader_net, spec,
                                             eval_rate_moves_sd, rule=rule,
                                             fee=MM_FEE)[:2]
                        for rule in rules}
        # a smaller fresh replay WITH the play recorded, for the diagnostics
        record_generator = torch.Generator().manual_seed(SEED_PATH + 3)
        record_rate_moves_sd = torch.randn(RECORD_PATHS, spec.rounds + 1,
                                           generator=record_generator)
        record_mm_pl, record_trader_pl, log = M.rollout(
            mm_net, trader_net, spec, record_rate_moves_sd, tau=None,
            fee=MM_FEE, record=True)
        record_bdc_pl = always_bdc_pl(spec, record_rate_moves_sd)
        # the demo game's rates (BdC_Calculations.ods, Game sheet), replayed
        # through the co-trained pair: the z-draws that land exactly on them
        set_rates = np.asarray(SET_PATH)
        set_moves = torch.tensor((set_rates - np.concatenate(
            ([spec.params.a0], set_rates[:-1]))) / spec.params.sd).unsqueeze(0)
        set_mm_pl, set_trader_pl, set_log = M.rollout(
            mm_net, trader_net, spec, set_moves, tau=None, fee=MM_FEE,
            record=True)
        set_bdc_pl = always_bdc_pl(spec, set_moves)
        p1_log = (M.rollout(p1_mm, p1_trader, spec, record_rate_moves_sd,
                            tau=None, fee=MM_FEE, record=True)[2]
                  if phase1 is not None else None)

    co_mm_pl, co_trader_pl = eval_pl["co MM v co trader"]
    benchmark_rule = max(rules, key=lambda r: float(eval_rule_pl[r.name][0].mean()))
    agree, filled, filled_benchmark, benchmark_name = M.volume_agreement(
        log, side, benchmark_rule, fee=MM_FEE)
    volume = log["offered_amount"]
    price_over_rate = log["quoted_price"] / log["hidden_rate"]
    threshold_by_round = [
        M.operating_threshold(price_over_rate[:, r], volume[:, r],
                              log["fill_weight"][:, r], side)[0]
        if volume[:, r].sum() > 0 else np.nan for r in range(spec.rounds)]
    repay_by_round = M.repay_on_short_books(log)

    def pct(x):
        return f"{float(x.mean()) * 100:+.4f}%"

    def bp(x):
        return f"{float(x.mean()) * 1e4:+.2f} bp"

    two_se = 2 * float(co_mm_pl.std()) / np.sqrt(EVAL_PATHS)
    edge, edge_2se = M.paired(co_trader_pl, eval_bdc_pl)
    print(f"\nside {side} final (hard game, {EVAL_PATHS:,} paths, torch engine):")
    print(f"  co-trained MM P/L {bp(co_mm_pl)} +/- {two_se * 1e4:.2f} of the "
          f"game's capital (GBP {M.game_capital_pounds(spec):,.0f});  "
          f"co-trained trader P/L {pct(co_trader_pl)}")
    print(f"  trader edge over always-BdC {edge * 100:+.4f}% +/- "
          f"{edge_2se * 100:.4f}% (paired on these paths; always-BdC here "
          f"{pct(eval_bdc_pl)}, analytic {bdc_floor * 100:+.4f}%)")
    for rule in rules:
        rule_mm_pl, rule_trader_pl = eval_rule_pl[rule.name]
        gap, gap_2se = M.paired(co_mm_pl, rule_mm_pl)
        print(f"  vs {rule.name:<18} MM {bp(rule_mm_pl):>10}   co MM - rule "
              f"{gap * 1e4:+.2f} +/- {gap_2se * 1e4:.2f} bp (paired)   "
              f"trader vs this rule {pct(rule_trader_pl)}")
    print(f"  verdicts agree with the {benchmark_name} rule on {agree * 100:.1f}% "
          f"of offered volume; filled {filled * 100:.1f}% "
          f"({benchmark_name} {filled_benchmark * 100:.1f}%)")
    print("  operating threshold P/X by round, fitted to the verdicts: "
          + "  ".join(f"r{r + 1} {threshold_by_round[r]:.4f}"
                      for r in range(spec.rounds)))
    print("  Bureau repay fraction on SHORT books, short-weighted, by round: "
          + "  ".join(f"r{r + 1} {repay_by_round[r]:.3f}"
                      for r in range(spec.rounds)))
    print(f"  round-1 quote, $/GBP (median): co-trained "
          f"{np.median(log['quoted_price'][:, 0]):.4f}"
          + (f"   Phase 1 {np.median(p1_log['quoted_price'][:, 0]):.4f}"
             if p1_log is not None else ""))

    cross = {}
    if phase1 is not None:
        # THE EQUILIBRIUM TEST. Rows: the deviating agent; each line is the
        # deviation's gain, paired on the same paths. A best response gains
        # nothing: every line should be <= 0 within its 2 SE
        p1_mm_pl, p1_trader_pl = eval_pl["P1 MM v P1 trader"]
        x_mm_pl, x_trader_pl = eval_pl["P1 MM v co trader"]   # P1 MM deviates
        y_mm_pl, y_trader_pl = eval_pl["co MM v P1 trader"]   # P1 trader deviates
        cross = {
            # against the CO-TRAINED partner: is the co-trained pair stable?
            "P1 MM replacing co MM (v co trader), MM gain":
                M.paired(x_mm_pl, co_mm_pl),
            "P1 trader replacing co trader (v co MM), trader gain":
                M.paired(y_trader_pl, co_trader_pl),
            # against the PHASE 1 partner: was Phase 1's pair stable too?
            "co MM replacing P1 MM (v P1 trader), MM gain":
                M.paired(y_mm_pl, p1_mm_pl),
            "co trader replacing P1 trader (v P1 MM), trader gain":
                M.paired(x_trader_pl, p1_trader_pl),
        }
        print(f"\n  cross-play with Phase 1's pair (trader {p1_fingerprint}), "
              f"same {EVAL_PATHS:,} paths:")
        print(f"    P1 MM v P1 trader   MM {bp(p1_mm_pl):>10}   trader {pct(p1_trader_pl)}")
        print(f"    co MM v co trader   MM {bp(co_mm_pl):>10}   trader {pct(co_trader_pl)}")
        print(f"    P1 MM v co trader   MM {bp(x_mm_pl):>10}   trader {pct(x_trader_pl)}")
        print(f"    co MM v P1 trader   MM {bp(y_mm_pl):>10}   trader {pct(y_trader_pl)}")
        print("  deviation gains (paired; a best response gains nothing, so "
              "each should be <= 0 within 2 SE):")
        for name, (gain, gain_2se) in cross.items():
            unit = 1e4 if "MM gain" in name else 100
            label = "bp" if unit == 1e4 else "%"
            print(f"    {name:<52} {gain * unit:+.3f} +/- {gain_2se * unit:.3f} {label}")

    # ---- 5. ARTEFACTS -----------------------------------------------------
    # everything cotrain_diagnostics needs, and nothing it must recompute
    np.savez(run / "cotrain_eval.npz",
             side=side, fee=MM_FEE, first=FIRST, mm_level_init=MM_LEVEL_INIT,
             n_outer=N_OUTER, n_inner=N_INNER, batch=BATCH,
             lr=np.array([LR_TRADER, LR_MM]), trader_plateau=TRADER_PLATEAU,
             tau_trader=np.array(TAU_TRADER), tau_mm=np.array(TAU_MM),
             tau_anneal_frac=TAU_ANNEAL_FRAC,
             card=np.array([spec.L, spec.T, spec.A, spec.B]),
             rounds=spec.rounds, k=spec.K,
             a0=spec.params.a0, sd=spec.params.sd,
             trader_bdc_fee=spec.params.bdc_fee,
             capital_pounds=M.game_capital_pounds(spec),
             bdc_floor=bdc_floor,                    # analytic (v2)
             eval_bdc_pl=eval_bdc_pl.numpy(),        # per path: the floor,
             record_bdc_pl=record_bdc_pl.numpy(),    # paired with each set
             set_bdc_pl=float(set_bdc_pl[0]),
             agree=agree, filled=filled, filled_benchmark=filled_benchmark,
             rule_names=np.array([rule.name for rule in rules]),
             rule_repay=np.array([rule.repay for rule in rules]),
             benchmark_name=benchmark_name, benchmark_rule=benchmark_rule.name,
             eval_names=np.array(list(eval_pl)),
             eval_mm_pl=np.stack([pl[0].numpy() for pl in eval_pl.values()]),
             eval_trader_pl=np.stack([pl[1].numpy() for pl in eval_pl.values()]),
             eval_rule_mm_pl=np.stack([eval_rule_pl[r.name][0].numpy() for r in rules]),
             eval_rule_trader_pl=np.stack([eval_rule_pl[r.name][1].numpy() for r in rules]),
             cross_names=np.array(list(cross)),
             cross_gain=np.array([g for g, _ in cross.values()]),
             cross_2se=np.array([s for _, s in cross.values()]),
             threshold_by_round=np.array(threshold_by_round),
             repay_by_round=np.array(repay_by_round),
             phase1_fingerprint=(p1_fingerprint if phase1 is not None else ""),
             record_mm_pl=record_mm_pl.numpy(),
             record_trader_pl=record_trader_pl.numpy(),
             set_path=np.array(SET_PATH),
             set_mm_pl=float(set_mm_pl[0]), set_trader_pl=float(set_trader_pl[0]),
             **{f"set_record_{key}": value for key, value in set_log.items()},
             **{f"record_{key}": value for key, value in log.items()},
             **({f"p1_record_{key}": value for key, value in p1_log.items()}
                if p1_log is not None else {}))
    torch.save({"trader": trader_net.state_dict(), "mm": mm_net.state_dict()},
               run / "pair_final.pth")
    print(f"artefacts in {run}/  ({(time.time() - start_time) / 60:.2f} min)",
          flush=True)
    return {"side": side, "mm": float(co_mm_pl.mean()),
            "trader": float(co_trader_pl.mean()), "cross": cross,
            "threshold": threshold_by_round, "curve": curve}


def main():
    summaries = [run_side(side) for side in SIDES]
    print("\n" + "=" * 72)
    print(f"CO-TRAINED PAIRS, fee {MM_FEE}, hard game, {EVAL_PATHS:,} paths per "
          f"side")
    for s in summaries:
        worst = max((g for g, _ in s["cross"].values()), default=float("nan"))
        print(f"  side {s['side']}: MM {s['mm'] * 1e4:+8.2f} bp   trader "
              f"{s['trader'] * 100:+.4f}%   round-1 threshold "
              f"{s['threshold'][0]:.4f} x true rate   largest deviation gain "
              f"{worst:+.2e}")
    print("  validation read: the pair should land where Phase 1 did (side A "
          "level ~ 0.99 x X and flatten; side B carry) AND no deviation "
          "should gain -- that second half is what Phase 1 could not show")


if __name__ == "__main__":
    main()