"""
score_best.py  --  verify the best-vs-final scoring gap. READ-ONLY.

For each run it loads BOTH the final model (what summary.txt currently reports)
and best_model.zip, and scores them with the SAME evaluator on the SAME
held-out seed set (123 -- the run_diagnostics default, which is separate from
EvalCallback's own eval seeds). So the numbers are directly comparable to the
existing summary.txt, with no winner's-curse bias.

Nothing is retrained; no files are overwritten. Run from the folder that holds
v2rl_env.py / rl_diagnostics.py, with the results/ tree present:

    python score_best.py

Decision rule: if  best_mean - 2*s.e.  >  always-BdC (-1.44%),  the run's best
policy genuinely clears the naive baseline -> the scoring mistake was material
and `best` is the corrected headline. If best ~= final, the run plateaued and
"RL does not beat BdC" stands (the unused best_model.zip was only cosmetic).
"""

import os
import math

from stable_baselines3 import PPO, DQN
from v2rl_env import V2Game, GridV2Game, GameSpec
from rl_diagnostics import evaluate, BDC_BASELINE_PCT, V2_DP_PCT

DIAG_GAMES = 5_000        # match the training scripts -> comparable to summary.txt
SEED       = 123          # held out from EvalCallback's eval seeds

# (run_dir, loader class, env factory, final-model filename)
# DoubleDQN checkpoints load fine with DQN here: .load() only rebuilds the
# policy for predict(), and DoubleDQN only overrides train()/target maths.
RUNS = [
    ("results/ppo_v2rl", PPO, lambda: V2Game(GameSpec()),     "ppo_final"),
    ("results/dqn_shaped", DQN, lambda: GridV2Game(GameSpec()), "dqn_final"),
    # ("results/dqn_shaped", DQN, lambda: GridV2Game(GameSpec()), "dqn_final"),
]


def score(model_path, klass, env_factory):
    model = klass.load(model_path)                       # predict only; no env needed
    pl = evaluate(model, env_factory, n_games=DIAG_GAMES, seed=SEED)["pl"] * 100.0
    return float(pl.mean()), float(2 * pl.std(ddof=1) / math.sqrt(len(pl)))


for run_dir, klass, env_factory, final_name in RUNS:
    print("=" * 64)
    print(run_dir)
    for label, name in [("final  (summary.txt reports this)", final_name),
                        ("best_model.zip", "best_model")]:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path + ".zip"):
            print(f"  {label:34}  MISSING  ({path}.zip)")
            continue
        m, se2 = score(path, klass, env_factory)
        verdict = "clears BdC" if m - se2 > BDC_BASELINE_PCT else "does NOT clear BdC"
        print(f"  {label:34}  {m:+7.3f}%  +/-{se2:.3f}   [{verdict}]")
    print(f"  reference:  always-BdC {BDC_BASELINE_PCT:+.3f}%    v2 DP {V2_DP_PCT:+.3f}%")