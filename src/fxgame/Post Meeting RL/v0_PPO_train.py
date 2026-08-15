"""
rl_train_ppo.py -- SB3 PPO on the certified environment (rl_env.Game).

A TRADER WITH NO PRIOR KNOWLEDGE. This arm represents someone sitting down to
play the game knowing only the rules, so it is held to a strict standard: no
quantity in this file is derived from the rate distribution, and no
hyperparameter is tuned against the known answer. Whatever it scores is the
result, including a bad one. PPO is stock; the env is used exactly as
certified, with no wrappers.
"""

# ============================== SETTINGS ======================================
# The GAME is not configured here -- card, rounds, K, side and fee are read
# from rl_env.Game().spec. Nothing below is derived from the rate distribution.
RUN_NAME       = "ppo_A_R4_f200bp"   # results/<RUN_NAME>/ ; rename per run
TOTAL_EPISODES = 300000   # budgeted at the MAXIMUM episode length; the env's
                           # stop action can only make episodes shorter
N_ENVS         = 8         # parallel envs: speed only, does not change the game
SEED           = 0
GAMMA          = 1.0       # scored once, at settlement
N_EVALS        = 20        # points on the learning curve
EVAL_EPISODES  = 2_000     # per evaluation (unpaired -- see the header)
REPORT         = True      # the paired replay + the five outputs
RESULTS_DIR    = "results"
MODEL_NAME     = "ppo_game"        # the name rl_diagnostics expects
# ==============================================================================

from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from rl_env import Game
import rl_diagnostics as D

OUT = Path(__file__).resolve().parent / RESULTS_DIR / RUN_NAME


def run_dir(spec):
    """rl_diagnostics.main() imports this to find the saved model."""
    return OUT


def wrap_action(spec, u, env):
    """Identity. PPO acts directly in the env's own action space, so there is
    nothing to translate -- kept only because rl_diagnostics.main() imports it
    to rebuild the policy when re-reporting without retraining."""
    return np.asarray(u, dtype=np.float32)


def make_env():
    return Game()


def main():
    set_random_seed(SEED)
    spec = Game().spec                       # read, never chosen here
    OUT.mkdir(parents=True, exist_ok=True)
    max_steps = spec.rounds * (spec.K + 1)
    total_timesteps = TOTAL_EPISODES * max_steps

    print(f"game (from rl_env): side {spec.side}, rounds {spec.rounds}, "
          f"K {spec.K}, fee {spec.params.bdc_fee:.2%}")
    print(f"  budget {TOTAL_EPISODES:,} episodes x <= {max_steps} steps = "
          f"{total_timesteps:,} timesteps")
    print("  stock PPO on the certified env: no wrappers, no tuning, no "
          "distribution knowledge")

    train_env = make_vec_env(make_env, n_envs=N_ENVS, seed=SEED)
    eval_env = make_vec_env(make_env, n_envs=1, seed=SEED + 999)
    freq = max(total_timesteps // (N_EVALS * N_ENVS), 1)

    eval_cb = EvalCallback(eval_env, best_model_save_path=str(OUT),
                           log_path=str(OUT), eval_freq=freq,
                           n_eval_episodes=EVAL_EPISODES, deterministic=True,
                           verbose=0)
    ckpt_cb = CheckpointCallback(save_freq=freq,
                                 save_path=str(OUT / "checkpoints"),
                                 name_prefix="ppo")

    model = PPO("MlpPolicy", train_env, seed=SEED, gamma=GAMMA, verbose=0)
    model.learn(total_timesteps=total_timesteps,
                callback=[eval_cb, ckpt_cb], progress_bar=True)
    model.save(OUT / MODEL_NAME)

    ev = np.load(OUT / "evaluations.npz")
    xs, ys = ev["timesteps"], ev["results"].mean(axis=1)
    print(f"  eval curve {ys[0] * 100:+.3f}% -> {ys[-1] * 100:+.3f}% "
          f"(unpaired, +/-~1% -- shape only)")

    if REPORT:
        pol = lambda obs, env: model.predict(obs, deterministic=True)[0]
        D.report(spec, pol, OUT, label="PPO", curve=(xs, ys, "timestep"))


if __name__ == "__main__":
    main()