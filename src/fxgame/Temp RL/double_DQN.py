"""
================================================================================
 train_dqn.py  --  train ONE DQN agent on v2's OWN discrete action grid
================================================================================

 Run once:  python train_dqn.py

 The point of this file: DQN plays on exactly the action set the v2 DP optimises
 over (v2rl_env.GridV2Game -- the 49-price x quantity grid + stop), so the
 learned policy is directly comparable to the DP. Same mechanics, observation
 and reward as the PPO env; only the action interface is discrete.

 To change a run, edit the SETTINGS block below -- there is no command line.
 Everything lands under results/<RUN_NAME>/ (see README.txt for the file list).
================================================================================
"""

import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from v2rl_env import GridV2Game, GameSpec
from rl_common import RewardScale, PotentialShaping, DoubleDQN
import rl_diagnostics as diag

# ==============================================================================
# SETTINGS -- edit these
# ==============================================================================
RUN_NAME      = "dqn_shaped"   # results land in results/<RUN_NAME>/
SHAPED        = True           # dense potential-based reward (optimum provably unchanged);
                               # False = the sparse terminal-only baseline
TIMESTEPS     = 2_000_000      # training budget (DQN needs more than PPO: 1,275 actions)
SEED          = 0
EVAL_EVERY    = 50_000         # timesteps between deterministic evals (the curve)
EVAL_EPISODES = 2_000          # episodes per eval point
DIAG_GAMES    = 5_000          # games for the final scorecard + figures
DOUBLE_DQN    = True           # Double-DQN target (unit-verified in rl_common);
                               # False = SB3's vanilla max-Q target
# ==============================================================================


def make_env():
    env = GridV2Game(GameSpec())
    if SHAPED:
        env = PotentialShaping(env)      # training-side only; diagnostics use the raw env
    return Monitor(RewardScale(env))


def main():
    out = os.path.join("results", RUN_NAME) # "results/dqn_v2rl"
    ckpt_dir = os.path.join(out, "checkpoints") # "results/dqn_v2rl/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    #Evaluates periodically the performance of the agent, using a separate test environment and saves the best model
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=out, log_path=out,
        eval_freq=EVAL_EVERY, n_eval_episodes=EVAL_EPISODES,
        deterministic=True, render=False, verbose = 0)
    
    #saves a model as a checkpoint
    ckpt_cb = CheckpointCallback(
        save_freq=EVAL_EVERY, save_path=ckpt_dir, name_prefix="dqn")

    # gamma=1.0 for the same finite-horizon reason as PPO. Hyperparameters are
    # standard DQN defaults for a task this size; tune later if the curve stalls.
    Algo = DoubleDQN if DOUBLE_DQN else DQN
    model = Algo(
        "MlpPolicy", train_env, seed=SEED, gamma=1.0,
        learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000,
        batch_size=128, train_freq=4, target_update_interval=2_000,
        exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.60, policy_kwargs={"net_arch": [256, 256]},
        verbose=0)
    model.learn(total_timesteps=TIMESTEPS, callback=[eval_cb, ckpt_cb],
                progress_bar=True)
    model.save(os.path.join(out, "dqn_final"))

    s = diag.run_diagnostics(model, out, env_factory=lambda: GridV2Game(GameSpec()),
                             n_games=DIAG_GAMES, algorithm="DQN")

    print("\n" + "=" * 52)
    print(f"{'strategy':<24}{'P/L':>10}   2 s.e.")
    print("-" * 52)
    print(f"{'DQN (this run)':<24}{s['mean_pct']:+8.3f}%  {s['se2_pct']:.3f}%")
    print(f"{'always-BdC (computed)':<24}{diag.BDC_BASELINE_PCT:+8.3f}%     --")
    print(f"{'T1 DP optimum (target)':<24}{diag.V2_DP_PCT:+8.3f}%     --")
    print("=" * 52)
    print(f"all results in {out}/")


if __name__ == "__main__":
    main()