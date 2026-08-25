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
import NN_one_offer_game.torch_train as T
from Mechanics.fx_mechanics import results_path

# THE TWO CARDS THIS STUDY PLAYS, stated in full. rl_env._card holds the
# game's own defaults and is CERTIFIED -- never edited -- so a study's card is
# written out here instead. It lives in this file alone because mm_train.py
# imports it: the frozen trader and the MM judging it must play the same game.
#   T1  GBP 100,000 -> USD 125,000  A 2% B 3%   T4  USD 500,000 -> GBP 396,000
#   T2  USD 200,000 -> GBP 155,000  A 3% B 2%       A 0% B 20%  (side B, cert.)
# Side B is T2, not T4: both penalties are live, which T4's 0% residue leaves
# untested.
CARD = {"A": GameSpec(L=100_000.0, T=125_000.0, A=0.02, B=0.03, side="A"),
        "B": GameSpec(L=200_000.0, T=155_000.0, A=0.03, B=0.02, side="B")}

SIDES = ("A", "B")        # both cards; DONE markers make re-runs skip -- but
                          # ONLY when the card and the trader's own settings
                          # are unchanged (see _trader_fingerprint below)


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
        "iters": T.ITERS, "batch": T.BATCH, "lr": T.LR,
        "seeds": [T.SEED_INIT, T.SEED_PATH],
        "src": hashlib.sha256("".join(
            inspect.getsource(f) for f in
            (T.AlphaNet, T.squash, T.features, T.rollout, T.train)
        ).encode()).hexdigest()[:16],
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main():
    for side in SIDES:
        spec = CARD[side]                        # stated at the top
        done_marker = results_path(f"mm_phase1/trader_{side}") / "DONE"
        fingerprint = _trader_fingerprint(spec)
        if done_marker.exists() and done_marker.read_text().strip() != fingerprint:
            print(f"side {side}: STALE trader -- the DONE fingerprint does not "
                  f"match the current card or trader settings. Discarding and "
                  f"retraining; reusing it would freeze a trader built for a "
                  f"DIFFERENT game.", flush=True)
            for stale in ("policy_best.pth", "policy_final.pth", "curve.npz"):
                (results_path(f"mm_phase1/trader_{side}") / stale).unlink(
                    missing_ok=True)
            done_marker.unlink()
        if done_marker.exists():
            print(f"side {side}: DONE fingerprint matches, skipping")
            continue
        print(f"\n################ side {side}: card L={spec.L:,.0f} "
              f"T={spec.T:,.0f} A={spec.A} B={spec.B} rounds={spec.rounds} "
              f"K={spec.K} ################", flush=True)
        T.check_torch_game_matches_env(spec)     # check 1: torch game == env
        T.check_policy_replay_matches_env(spec)  # check 2: policy seam == env
        run = results_path(f"mm_phase1/trader_{side}")
        run.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        net, curve = T.train(spec, run)          # torch_train settings, as-is
        np.savez(run / "curve.npz", iters=np.array(curve[0]),
                 hard_pl=np.array(curve[1]))
        # final large hard-game evaluation, torch engine (fresh, never trained)
        gen = torch.Generator().manual_seed(T.SEED_PATH + 2)
        with torch.no_grad():
            pl = T.rollout(net, spec,
                           torch.randn(T.EVAL_PATHS, spec.rounds + 1,
                                       generator=gen), tau=None)
        done_marker.write_text(fingerprint)
        print(f"side {side} trader: hard-game P/L "
              f"{float(pl.mean())*100:+.4f}% +/- "
              f"{2*float(pl.std())/np.sqrt(T.EVAL_PATHS)*100:.4f}%  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print("\nTRADER TRAINING DONE")


if __name__ == "__main__":
    main()