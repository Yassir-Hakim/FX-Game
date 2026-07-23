"""
================================================================================
 train_ppo.py  --  train ONE continuous PPO agent on the v2 game
================================================================================

 Run once:  python train_ppo.py

 Trains PPO on the continuous v2rl_env.V2Game, saves the model + learning curve
 + checkpoints, writes the diagnostic figures, prints a scorecard. The analysis
 notebook then RELOADS the saved model, so you never retrain to look again.

 To change a run, edit the SETTINGS block below -- there is no command line.
 Everything lands under results/<RUN_NAME>/ (see README.txt for the file list).
================================================================================
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from v2rl_env import V2Game, GameSpec
from rl_common import RewardScale
import rl_diagnostics as diag

# ==============================================================================
# SETTINGS
# ==============================================================================
RUN_NAME      = "ppo_v2rl"     # results land in results/<RUN_NAME>/
TIMESTEPS     = 1_000_000      # training budget
N_ENVS        = 8              # parallel envs (speed only; does not change the game)
SEED          = 0
EVAL_EVERY    = 50_000         # timesteps between deterministic evals (the curve)
EVAL_EPISODES = 2_000          # episodes per eval point
DIAG_GAMES    = 5_000          # games for the final scorecard + figures
# ==============================================================================


def make_env():
    return RewardScale(V2Game(GameSpec()))


def main():
    out = os.path.join("results", RUN_NAME) # "results/ppo_v2rl"
    ckpt_dir = os.path.join(out, "checkpoints") # "results/ppo_v2rl/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    #allows us to train it on n environments per step
    train_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED)
    eval_env = make_vec_env(make_env, n_envs=1, seed=SEED + 999)


    freq_per_env = max(EVAL_EVERY // N_ENVS, 1)

    #Evaluates periodically the performance of the agent, using a separate test environment and saves the best model
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=out, log_path=out,
        eval_freq=freq_per_env, n_eval_episodes=EVAL_EPISODES,
        deterministic=True, render=False, verbose = 0)
    
    #saves a model as a checkpoint
    ckpt_cb = CheckpointCallback(save_freq=freq_per_env, save_path=ckpt_dir, name_prefix="ppo")
    
    model = PPO("MlpPolicy", train_env, seed=SEED, gamma=1.0, verbose=0)
    model.learn(total_timesteps=TIMESTEPS, callback=[eval_cb, ckpt_cb],
                progress_bar=True)
    model.save(os.path.join(out, "ppo_final"))
 
    s = diag.run_diagnostics(model, out, env_factory=lambda: V2Game(GameSpec()),
                             n_games=DIAG_GAMES, algorithm="PPO")
 
    print("\n" + "=" * 52)
    print(f"{'strategy':<24}{'P/L':>10}   2 s.e.")
    print("-" * 52)
    print(f"{'PPO (this run)':<24}{s['mean_pct']:+8.3f}%  {s['se2_pct']:.3f}%")
    print(f"{'always-BdC (computed)':<24}{diag.BDC_BASELINE_PCT:+8.3f}%     --")
    print(f"{'T1 DP optimum (target)':<24}{diag.V2_DP_PCT:+8.3f}%     --")
    print("=" * 52)
    print(f"all results in {out}/")
 
 
if __name__ == "__main__":
    main()
