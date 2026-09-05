import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "NN_one_offer_game"      # the folder holding rl_env.py and
                                         # torch_train.py; edit this ONE name
                                         # if that folder is ever renamed
assert (_CORE / "rl_env.py").exists(), f"core folder not found: {_CORE}"
sys.path.insert(0, str(_CORE))           # the certified machinery layer
sys.path.insert(0, str(_ROOT))           # Mechanics/, DP_Models/
# ============================================================================
# train_traders.py -- driver for the FROZEN traders of the MM phase-1 study.
# Certifies torch_train on both cards (checks 1 and 2), then trains each
# trader with torch_train's OWN settings, untouched. The cards come from
# MM_Phase_1/cards.py, never from edits to any certified file (rl_env._card
# is the game's own default and stays untouched). DONE markers make
# finished sides skip, so the script is safe to re-run. Run it; never run
# torch_train.py directly for card results -- its main() trains the DEFAULT
# GameSpec, which is not a card.
# ============================================================================
import hashlib
import inspect
import json
import time

import numpy as np
import torch

from NN_one_offer_game.rl_env import GameSpec
import torch_train_fee as T              # the engine now carries the MM's
                                         # own Bureau cost (T.MM_FEE); at
                                         # T.MM_FEE = 0 it IS torch_train,
                                         # bit for bit
from Mechanics.fx_mechanics import GameParams, results_path

# THE TWO CARDS THIS STUDY PLAYS, stated in full. rl_env._card holds the
# game's own defaults and is CERTIFIED -- never edited -- so a study's card is
# written out here instead. It lives in this file alone because mm_train.py
# imports it: the frozen trader and the MM judging it must play the same game.
# The numbers now live in torch_train_fee.py's SETTINGS block (CARDS), edited
# there beside MM_FEE, BDC_FEE and ROUNDS; this file re-exports them, so
# mm_train.py's "from train_traders import CARD" is unchanged and there is
# still exactly ONE definition of the cards in the study.
CARD = {side: T.build_spec(side) for side in ("A", "B")}

SIDES = ("A", "B")
RECORD_PATHS = 20_000     # of the evaluation paths, replayed WITH a record
                          # for trader_diagnostics; P/L uses all EVAL_PATHS        # both cards; DONE markers make re-runs skip -- but
                          # ONLY when the card and the trader's own settings
                          # are unchanged (see _trader_fingerprint below)


def trader_run(side):
    """The folder a frozen trader lives in, tagged by the MM fee it was
    trained against, so corrected and naive artefacts never mix. mm_train
    imports this: it always faces the trader THIS driver currently builds,
    and no fee constant appears over there. (The original untagged
    trader_{side} folders are legacy: g=0 retrains them bit-identically
    into trader_{side}_g0.)"""
    return results_path(f"mm_phase1/trader_{side}_g{T.MM_FEE:g}")


def _trader_fingerprint(spec):
    """Everything that determines what a trained trader IS. A finished
    marker is valid only for the configuration that produced it: keying the
    skip on the marker's mere existence let an edited CARD silently reuse a
    trader trained for a different one, whose quotes are then off-card by a
    measurable margin and whose every downstream number is meaningless."""
    payload = json.dumps({
        "card": [spec.L, spec.T, spec.A, spec.B, spec.side, spec.rounds,
                 spec.K],
        "market": [spec.params.a0, spec.params.sd, spec.params.bdc_fee],
        "mm_fee": T.MM_FEE,
        "iters": T.ITERS, "batch": T.BATCH, "lr": T.LR,
        "seeds": [T.SEED_INIT, T.SEED_PATH],
        "src": hashlib.sha256("".join(
            inspect.getsource(f) for f in
            (T.AlphaNet, T.squash, T.features, T.rollout, T.train)
        ).encode()).hexdigest()[:16],
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _save_eval(run, spec, net, pl, eval_moves):
    """Everything trader_diagnostics.py draws, written once here: a recorded
    replay of the frozen trader against the fee-aware rule at T.MM_FEE (the
    game it was trained on -- certified equal to T.rollout by check 1F) and
    against the naive rule for reference, plus the DP's always-Bureau floor.
    RECORD_PATHS games recorded, the P/L from the full T.EVAL_PATHS."""
    import mm_train  # deferred: mm_train imports this file
    from DP_Models.v2_multiple_rounds import bdc_baseline
    rules = {rule.name: rule for rule in mm_train.reference_rules(T.MM_FEE)}
    fee_rule = rules.get("fee-aware+flatten", rules["naive+flatten"])
    record_moves = eval_moves[:RECORD_PATHS]
    with torch.no_grad():
        _, _, log = mm_train.rollout(None, net, spec, record_moves,
                                     rule=fee_rule, fee=T.MM_FEE, record=True)
        naive_pl = mm_train.rollout(None, net, spec, eval_moves,
                                    rule=rules["naive+flatten"],
                                    fee=T.MM_FEE)[1]
    np.savez(run / "trader_eval.npz",
             side=spec.side, mm_fee=T.MM_FEE, bdc_fee=spec.params.bdc_fee,
             a0=spec.params.a0, L=spec.L, rounds=spec.rounds,
             eval_paths=T.EVAL_PATHS, record_paths=RECORD_PATHS,
             pl=pl.numpy(), pl_vs_naive=naive_pl.numpy(),
             bdc_floor=bdc_baseline(spec)[1], rule=fee_rule.name,
             **{f"record_{key}": value for key, value in log.items()})


def main():
    for side in SIDES:
        spec = CARD[side]                        # stated at the top
        done_marker = trader_run(side) / "DONE"
        fingerprint = _trader_fingerprint(spec)
        if done_marker.exists() and done_marker.read_text().strip() != fingerprint:
            print(f"side {side}: STALE trader -- the DONE fingerprint does not "
                  f"match the current card or trader settings. Discarding and "
                  f"retraining; reusing it would freeze a trader built for a "
                  f"DIFFERENT game.", flush=True)
            for stale in ("policy_best.pth", "policy_final.pth", "curve.npz",
                          "trader_eval.npz"):
                (trader_run(side) / stale).unlink(
                    missing_ok=True)
            done_marker.unlink()
        if done_marker.exists():
            print(f"side {side}: DONE fingerprint matches, skipping")
            continue
        print(f"\n################ side {side}: card L={spec.L:,.0f} "
              f"T={spec.T:,.0f} A={spec.A} B={spec.B} rounds={spec.rounds} "
              f"K={spec.K} ################", flush=True)
        if T.MM_FEE == 0.0:
            T.check_torch_game_matches_env(spec)     # check 1: torch game == env
            T.check_policy_replay_matches_env(spec)  # check 2: policy seam == env
        else:
            print(f"  checks 1-2 (vs rl_env) skipped: the certified env has no "
                  f"MM fee, so at MM_FEE={T.MM_FEE:g} it is a different game "
                  f"by construction. Check 1F replaces them.")
            T.check_game_matches_fee_rule(spec)      # check 1F: == mm_train's rule
        run = trader_run(side)
        run.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        net, curve = T.train(spec, run)          # torch_train settings, as-is
        np.savez(run / "curve.npz", iters=np.array(curve[0]),
                 hard_pl=np.array(curve[1]))
        # final large hard-game evaluation, torch engine (fresh, never trained)
        gen = torch.Generator().manual_seed(T.SEED_PATH + 2)
        eval_moves = torch.randn(T.EVAL_PATHS, spec.rounds + 1, generator=gen)
        with torch.no_grad():
            pl = T.rollout(net, spec, eval_moves, tau=None)
        _save_eval(run, spec, net, pl, eval_moves)
        done_marker.write_text(fingerprint)
        print(f"side {side} trader: hard-game P/L "
              f"{float(pl.mean())*100:+.4f}% +/- "
              f"{2*float(pl.std())/np.sqrt(T.EVAL_PATHS)*100:.4f}%  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print("\nTRADER TRAINING DONE")


if __name__ == "__main__":
    main()