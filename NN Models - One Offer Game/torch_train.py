import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
  Direct policy optimisation through a differentiable simulator: pathwise
  gradients through the game's rules and the hidden rate. Neural stochastic control.

MEASUREMENT (project standard, not learner features)
  Check 1 aborts the run unless this file's game equals the certified
  rl_env.Game to machine precision -- the one risk a reimplementation
  creates. All reported numbers come from replaying the trained policy
  through that certified env via rl_diagnostics.report() (five outputs,
  floor-anchored values). The learner is told nothing by either.

DISTRIBUTION KNOWLEDGE: none. No setting encodes a0 or sd.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch

from rl_env import Game, P_MIN, P_MAX
import rl_diagnostics as D
from Mechanics.fx_mechanics import results_path

# ============================ SETTINGS ======================================
GATE = "smoothed"     # "smoothed": annealed sigmoid (the recipe, made to
                      #             work on a jumping payoff)
                      # "hard":     hard MM acceptance for training
ITERS = 2000          # training iterations, one fresh Monte Carlo batch each
BATCH = 1000          # paths per iteration -- "a 1000 samples of the noise"
LR = 1e-3             # AdamW, with the repository's plateau decay on top
CLIP_NORM = 1.0       # cap on how far one step can move the weights, so a freak batch can't wreck the policy.
TAU_START = 1.0       # begin with a ramp one rate-move wide.
TAU_ANNEAL_FRAC = 0.35  # anneal tau over the FIRST this-fraction of training,
                      # then HOLD at TAU_END. Without it the schedule stretches
                      # with ITERS, so a longer run spends longer in the soft
                      # game -- where fractional fills pay out at any price, so
                      # "offer a lot and keep the rest" scores well. Measured:
TAU_END = 0.02        # end at one-fiftieth of that, essentially the true rule.
VAL_EVERY = 50        # hard-game validation cadence; best checkpoint kept
VAL_PATHS = 20_000
EVAL_PATHS = 200_000  # final hard-game evaluation (torch engine)
LAGS = 4              # how many recent rate MOVES the network sees (0 = off).
CHECKS = True         # check 1: this file's game must equal the certified env
SEED_INIT = 42        # network init (repository convention: fork_rng)
SEED_PATH = 1234     # path noise, independent of the init seed
# ============================================================================

torch.set_default_dtype(torch.float64)   

# ---------------------------------------------------------------------------
# 1. models -- ported from the repository's src/utils.py
# ---------------------------------------------------------------------------
class ResidualBlock(torch.nn.Module):
 
    def __init__(self, dims):
        super().__init__()                          
        self.fc1 = torch.nn.Linear(dims, dims)
        self.fc2 = torch.nn.Linear(dims, dims)
        self.act1 = torch.nn.GELU()
        self.act2 = torch.nn.GELU()
 
    def forward(self, x):
        identity = x
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        return h + identity
 
 
class AlphaNet(torch.nn.Module):
    #The policy network: 9 numbers in, 2 raw numbers out
    def __init__(self, n_in=None, dims=64, blocks=2, n_out=2):
        # N_FEAT = 5 + LAGS, so changing LAGS in SETTINGS resizes the input layer
        n_in = N_FEAT if n_in is None else n_in
        super().__init__()

        # First projection (n_in → dims)
        self.fc_in = torch.nn.Linear(n_in, dims)
        self.act_in = torch.nn.GELU()

        # Residual blocks
        self.blocks = torch.nn.ModuleList(
            ResidualBlock(dims) for _ in range(blocks))
        
        # Shortcut projection (n_in → dims)
        self.shortcut = torch.nn.Linear(n_in, dims)

        # collapse 64 -> 2. No activation here: the bounding happens in
        # squash(), where it can be aimed at this game's own limits.
        self.head = torch.nn.Linear(dims, n_out)
 
    def forward(self, x):
        # x: (n_games, N_FEAT) -- one row per game in the batch

        identity = self.shortcut(x)
        # input projection
        hidden = self.act_in(self.fc_in(x))    

        # pass through residual blocks   
        for block in self.blocks:
            hidden = block(hidden)               

        hidden = hidden + identity

        return self.head(hidden) 


def squash(raw, anchor):
    #Turn the head's two unbounded numbers into a legal action
    price = 2.0 * anchor * torch.sigmoid(raw[..., 0]) #factor 2 makes prices above the anchor reachable
    frac = torch.sigmoid(raw[..., 1])
    return price, frac


def build_net(seed):
    #Create a fresh, randomly-initialised network.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return AlphaNet()


# ---------------------------------------------------------------------------
# 2. the game, explicit in torch ops
# ---------------------------------------------------------------------------
N_FEAT = 5 + LAGS

# The nine numbers the trader can see, assembled into one row per game.
def features(rounds_remaining_fraction, phase_flag, held_amount,
             banked_amount, last_revealed_rate, spec, recent_moves=None):

    n_games = held_amount.shape[0]

    def as_column(value):
        return (torch.full((n_games,), float(value))
                if np.isscalar(value) else value)

    feature_columns = [
        as_column(rounds_remaining_fraction),  # 1  in [0,1], NOT the raw round
        as_column(phase_flag),                 # 2  0 = offering, 1 = at the BdC
        held_amount / spec.L,                  # 3  fraction of the opening book
        banked_amount / max(spec.T, 1.0),      # 4  fraction of the target
        last_revealed_rate,                    # 5  the anchor, absolute $/GBP
    ]
    if LAGS:
        if recent_moves is None:               # round 1: no history yet
            recent_moves = torch.zeros(n_games, LAGS)
        # 6..  the last LAGS rate MOVES, newest first, zero-padded early
        feature_columns += [recent_moves[:, j] for j in range(LAGS)]
    return torch.stack(feature_columns, dim=-1)

# The game itself, re-written in torch so autodiff can see through it.
# Vectorised for efficiancy returns one p/l per game
def rollout(net, spec, rate_moves_sd, tau=None, forced=None, tau_abs=False):
    market = spec.params
    n_games, n_rounds, side = rate_moves_sd.shape[0], spec.rounds, spec.side
    
    # the pound-SELLER is filled when the hidden rate is ABOVE the quote, the buyer when below
    fill_direction = 1.0 if side == "A" else -1.0

    held_amount = torch.full((n_games,), float(spec.L))   # start currency
    banked_amount = torch.zeros(n_games)                  # target currency
    last_revealed_rate = torch.full((n_games,), float(market.a0))

    # the last LAGS rate moves the trader has watched, newest first
    recent_moves = torch.zeros(n_games, LAGS) if LAGS else None

    # the whole hidden rate path for every game, drawn up front. These are the
    # SIMULATOR's variables -- the network is never shown them.
    hidden_rates = market.a0 + torch.cumsum(
        market.sd * rate_moves_sd[:, :n_rounds], dim=1)
    settlement_rate = hidden_rates[:, -1] + market.sd * rate_moves_sd[:, n_rounds]

    for round in range(n_rounds):
        rounds_remaining_fraction = (n_rounds - round) / n_rounds
        # this round's hidden rate: HIDDEN during the offer, revealed after
        hidden_rate_now = hidden_rates[:, round]
        # ---- offer phase (K = 1, "one guess per round") --------------------
        if forced is not None:
            quoted_price = forced[round][0]
            offered_fraction = forced[round][1]
        else:
            quoted_price, offered_fraction = squash(
                net(features(rounds_remaining_fraction, 0.0, held_amount,
                             banked_amount, last_revealed_rate, spec,
                             recent_moves)),
                last_revealed_rate)
        offered_amount = offered_fraction * held_amount
        if tau is None:
            # THE REAL RULE: the market maker either accepts or does not.
            # 1.0 or 0.0, nothing between. Used for all evaluation.
            fill_weight = (fill_direction
                           * (hidden_rate_now - quoted_price) > 0).double() #.double() converts t/f to 1/0
        else:
            # THE TRAINING SUBSTITUTE: the same rule softened into a ramp, so
            # the payoff has a slope and autodiff has something to follow.
            # tau is the ramp's width, in units of one typical rate move --
            # measured from THIS batch
            move_scale = ((hidden_rate_now - last_revealed_rate)
                          .detach().std().clamp_min(1e-12))
            ramp_width = tau if tau_abs else tau * move_scale
            #MMs verdict on the offer
            fill_weight = torch.sigmoid(
                fill_direction * (hidden_rate_now - quoted_price) / ramp_width)
        if LAGS:
            # the rate is revealed at this point in the round, so the trader
            # now knows how far it moved: push that onto the history window
            recent_moves = torch.cat(
                [(hidden_rate_now - last_revealed_rate).detach().unsqueeze(1),
                 recent_moves[:, :-1]], dim=1)
        else:
            recent_moves = None

        filled_amount = fill_weight * offered_amount
        held_amount = held_amount - filled_amount
        banked_amount = banked_amount + (filled_amount * quoted_price
                                         if side == "A"
                                         else filled_amount / quoted_price)
        # ---- Bureau phase: rate revealed, optional conversion --------------
        if forced is not None:
            bdc_fraction = forced[round][2]
        else:
            _, bdc_fraction = squash(
                net(features(rounds_remaining_fraction, 1.0, held_amount,
                             banked_amount, hidden_rate_now, spec,
                             recent_moves)),
                hidden_rate_now)
        converted_amount = bdc_fraction * held_amount
        bdc_rate_after_fee = (1.0 - market.bdc_fee) * (
            hidden_rate_now if side == "A" else 1.0 / hidden_rate_now)
        held_amount = held_amount - converted_amount
        banked_amount = banked_amount + converted_amount * bdc_rate_after_fee

        # the revealed rate becomes next round's anchor
        last_revealed_rate = hidden_rate_now

    # ---- settlement (v2's terminal equation; verified in check 1) ----------
    # convert the banked currency back into the starting currency to score it:
    # side A banked dollars and divides by the rate, side B banked pounds and
    # multiplies. 
    settlement_factor = (1.0 / settlement_rate if side == "A"
                         else settlement_rate)
    shortfall = torch.clamp(spec.T - banked_amount, min=0.0)
    final_wealth = (held_amount * (1.0 - spec.A)                 # holding fee
                    + (banked_amount - spec.B * shortfall)       # target miss
                    * settlement_factor)
    return (final_wealth - spec.L) / spec.L


def pl_loss(net, spec, rate_moves_sd, tau):
    #negation is because optimisers MINIMISE and we want to MAXIMISE the payoff
    return -rollout(net, spec, rate_moves_sd, tau=tau).mean()


# ---------------------------------------------------------------------------
# 3. check 1 -- the one risk this file creates, discharged before training
# ---------------------------------------------------------------------------
def check_torch_game_matches_env(spec, n_games=400, seed=3):
    """Prove this file's torch game IS the certified game.

    The argument: play the SAME games with the SAME actions through both
    engines and require identical per-episode P/L. 

    Runs BEFORE training and raises on failure, because a trainer built on a
    subtly wrong copy of the game would optimise beautifully and mean nothing.
    """
    rng = np.random.default_rng(seed)
    n_rounds = spec.rounds

    # one forced rate path per game: rounds 1..R plus the settlement rate
    forced_rates = spec.params.a0 + spec.params.sd * np.cumsum(
        rng.standard_normal((n_games, n_rounds + 1)), axis=1)
    # random legal actions, one per half-step (offer and Bureau each round)
    random_prices = rng.uniform(P_MIN + 1e-6, P_MAX - 1e-6,
                                (n_games, 2 * n_rounds))
    random_fractions = rng.uniform(0.0, 1.0, (n_games, 2 * n_rounds))
    # the fractions cover the POLICY'S reachable set on THIS side: squash
    # emits magnitudes in (0,1), and torch_policy signs them by the side's
    # convention (side A spends the pound pocket with +, side B spends the
    # dollar pocket with -). rollout always trades the HELD pocket, so it
    # takes the magnitudes; the env reads the sign, so it gets them signed.
    env_sign = 1.0 if spec.side == "A" else -1.0

    # --- engine 1: the certified environment, one episode at a time --------
    env = Game(spec)
    env_pl = np.empty(n_games)
    for i in range(n_games):
        env.reset(options={"path": list(forced_rates[i, :n_rounds]),
                           "a_end": float(forced_rates[i, n_rounds])})
        done, step = False, 0
        while not done:
            action = np.array([random_prices[i, step],
                               env_sign * random_fractions[i, step]],
                              dtype=np.float64)
            _, reward, done, _, _ = env.step(action)
            step += 1
        env_pl[i] = reward          # the env pays the whole P/L at the end
    # --- engine 2: this file's torch rollout, all games at once ------------
    # rollout() takes standard-normal shocks, not rates, so invert: each
    # shock is (this rate - the previous one) / sd.
    previous_rates = np.concatenate(
        [np.zeros((n_games, 1)), forced_rates[:, :-1] - spec.params.a0],
        axis=1)
    shocks = torch.tensor((forced_rates - spec.params.a0 - previous_rates)
                          / spec.params.sd)
    # regroup the flat action list into per-round (price, offer, bureau)
    forced_actions = [(torch.tensor(random_prices[:, 2 * k]),
                       torch.tensor(random_fractions[:, 2 * k]),
                       torch.tensor(random_fractions[:, 2 * k + 1]))
                      for k in range(n_rounds)]
    torch_pl = rollout(None, spec, shocks, tau=None,
                       forced=forced_actions).numpy()

    # --- they must agree to floating-point noise, not to a tolerance -------
    max_gap = float(np.max(np.abs(torch_pl - env_pl)))
    passed = max_gap < 1e-9
    print(f"  check 1  torch game vs rl_env, {n_games} forced games "
          f"(side {spec.side}): max |P/L gap| {max_gap:.2e}  "
          f"[{'ok' if passed else 'FAIL'}]")
    assert passed, "the torch game does not match the certified env"


def check_policy_replay_matches_env(spec, n_paths=800, seed=7):
    """check 2 -- a NET-DRIVEN game must agree across engines.

    check 1 forces the actions, so it can never catch a bug in the seam it
    skips: features/squash/torch_policy -- the path a REAL policy takes.
    (Measured example: the side-B sign bug lived exactly there and check 1
    passed throughout.) Here an untrained network -- agreement is the claim,
    skill is not -- plays the same paths through rollout() and through the
    certified env via torch_policy(), and the per-game P/L must match.
    Tolerance 1e-6, not 1e-9: the env's observation is float32, so the
    policy reads the anchor at ~7 significant figures and the two engines
    legitimately differ in the 8th (measured ~1.6e-8).
    """
    net = build_net(SEED_INIT)
    X, a_end = D.draw_paths(spec, n_paths, seed)
    a0, sd = spec.params.a0, spec.params.sd
    previous = np.concatenate([np.full((n_paths, 1), a0), X[:, :-1]], axis=1)
    moves = np.concatenate([(X - previous) / sd,
                            ((a_end - X[:, -1]) / sd)[:, None]], axis=1)
    with torch.no_grad():
        torch_pl = rollout(net, spec, torch.tensor(moves), tau=None).numpy()
    _, _, env_pl, _ = D.run_paths(spec, torch_policy(net, spec), X, a_end,
                                  progress_every=0)
    max_gap = float(np.max(np.abs(torch_pl - env_pl)))
    passed = max_gap < 1e-6
    print(f"  check 2  net-driven, rollout vs env replay, {n_paths} games "
          f"(side {spec.side}): max |P/L gap| {max_gap:.2e}  "
          f"[{'ok' if passed else 'FAIL'}]")
    assert passed, "torch_policy does not reproduce rollout through the env"


# ---------------------------------------------------------------------------
# 4. training -- initialise, sample, average, backward, step
# --------------------------------------------------------------------------
def train(spec, run):
    net = build_net(SEED_INIT)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LR)
    # the repository's scheduler: if the loss stops improving, halve the
    # step size rather than keep overshooting
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=200)
    
    n_anneal = max(int(TAU_ANNEAL_FRAC * ITERS), 1)
    decay = (TAU_END / TAU_START) ** (1.0 / max(n_anneal - 1, 1))
    
    # the VALIDATION set: one fixed batch of paths, reused at every check so
    # checkpoints are compared on identical games. Its own seed, so it never
    # overlaps the training draws below.
    validation_generator = torch.Generator().manual_seed(SEED_PATH + 1)
    val_rate_moves_sd = torch.randn(VAL_PATHS, spec.rounds + 1,
                                   generator=validation_generator)
    best_checkpoint = {"hard_pl": -1e9, "iteration": -1}
    curve_iterations, curve_hard_pl = [], []
    for iteration in range(ITERS):
        # ramp width for this step: anneal, then hold (see TAU_ANNEAL_FRAC)
        if GATE == "hard":
            tau = None
        elif iteration < n_anneal:
            tau = TAU_START * decay ** iteration      # shrinking
        else:
            tau = TAU_END                             # held

        # STEP 1: draw a FRESH batch of games. A new seed every iteration, so
        # no game is ever seen twice and there is nothing to memorise.
        batch_generator = torch.Generator().manual_seed(
            SEED_PATH + 10_000 + iteration)
        batch_rate_moves_sd = torch.randn(BATCH, spec.rounds + 1,
                                  generator=batch_generator)

        # STEP 2: play them all with the current weights and average the P/L
        loss = pl_loss(net, spec, batch_rate_moves_sd, tau)
        if not torch.isfinite(loss):        
            print(f"  [guard] loss non-finite at iter {iteration}. "
                  f"Skipping step.")
            optimiser.zero_grad()
            continue

        # STEP 3: backprop
        optimiser.zero_grad()
        loss.backward()

        # STEP 4: cap the step's size, then take it
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_NORM)
        optimiser.step()
        lr_scheduler.step(loss.item())

        # STEP 5: periodically score the policy on the TRUE game (tau=None),
        # on the fixed validation paths, and keep the best version so far.
        if iteration % VAL_EVERY == 0 or iteration == ITERS - 1:
            with torch.no_grad():
                hard_game_pl = float(
                    rollout(net, spec, val_rate_moves_sd, tau=None).mean())
            curve_iterations.append(iteration)
            curve_hard_pl.append(hard_game_pl)
            tau_label = "  hard " if tau is None else f"tau {tau:8.5f}"

            # 'batch P/L' is a 1,000-path estimate of the SMOOTHED game and
            # is noisy by design; 'hard P/L' is 20,000 paths of the real one
            print(f"  iter {iteration:>5}  {tau_label}  "
                  f"batch P/L {-loss.item():+.5f}  "
                  f"hard P/L {hard_game_pl:+.5f}")
            if hard_game_pl > best_checkpoint["hard_pl"]:
                best_checkpoint.update(hard_pl=hard_game_pl,
                                       iteration=iteration)
                torch.save(net.state_dict(), run / "policy_best.pth")
    torch.save(net.state_dict(), run / "policy_final.pth")

    # early stopping: evaluate the checkpoint that validated best on the
    # HARD game -- the late anneal is the fragile stage (see header, item 3)
    print(f"  best hard-game validation "
          f"{best_checkpoint['hard_pl'] * 100:+.4f}% at iter "
          f"{best_checkpoint['iteration']}; evaluating policy_best.pth "
          f"(final kept on disk)")
    net.load_state_dict(torch.load(run / "policy_best.pth"))
    return net, (curve_iterations, curve_hard_pl, "iteration")


def torch_policy(net, spec):
    """The trained net as an env-side policy: rl_diagnostics replays it
    through the CERTIFIED env, so this arm and the PPO arm are measured by
    the same engine. The lag window is rebuilt here from the rates the ENV
    reveals (obs[6] == obs[7] at a Bureau step), never from env internals --
    the same information a student watching the game would have."""
    n_rounds = spec.rounds
    # the lag window has to be rebuilt as the episode is replayed, because
    # the env hands over one step at a time. Kept in a closure dict so it
    # survives between calls but resets cleanly at each new episode.
    history = {"moves": [0.0] * LAGS,      # newest first, same as in rollout
               "previous_rate": None,      # the last rate we saw revealed
               }

    def pol(obs, env):
        # at the OFFER phase the anchor is the last revealed rate (env.a);
        # at the BUREAU phase the rate has just been revealed (env.X)
        anchor_rate = float(env.a if env.phase == 0 else env.X)
        held_amount = float(env.c if spec.side == "A" else env.d)
        banked_amount = float(env.d if spec.side == "A" else env.c)

        if env.n == 1 and env.phase == 0:          # first step of an episode
            history.update(moves=[0.0] * LAGS, previous_rate=float(env.a))
        if env.phase == 1 and LAGS:
            # obs[6] and obs[7] are the bracket bounds on X; at a Bureau step
            # they collapse to the revealed rate itself. Read from the
            # OBSERVATION, not from env internals -- this is information the
            # trader is actually given.
            revealed_rate = float(obs[6])
            if history["previous_rate"] is not None:
                history["moves"] = (
                    [revealed_rate - history["previous_rate"]]
                    + history["moves"])[:LAGS]
            history["previous_rate"] = revealed_rate

        recent_moves = (torch.tensor([history["moves"]], dtype=torch.float64)
                        if LAGS else None)
        # (n_rounds - env.n + 1)/n_rounds: env.n counts 1..R, rollout counts
        # 0..R-1, so this matches rollout's rounds_remaining_fraction
        network_inputs = features((n_rounds - env.n + 1) / n_rounds,
                                  float(env.phase),
                                  torch.tensor([held_amount]),
                                  torch.tensor([banked_amount]),
                                  torch.tensor([anchor_rate]), spec,
                                  recent_moves)
        with torch.no_grad():                      # inference only, no graph
            quoted_price, fraction = squash(net(network_inputs),
                                            torch.tensor([anchor_rate]))
        # the env's action format: [price, signed fraction]. The SIGN picks
        # the pocket being spent: + spends env.c (pounds), - spends env.d
        # (dollars). The side-A trader holds pounds, the side-B trader holds
        # dollars, so the sign follows the side. At a Bureau step the env
        # ignores the price slot.
        signed = fraction if spec.side == "A" else -fraction
        return np.array([float(quoted_price[0]), float(signed[0])], dtype=np.float64)
    return pol


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
def main():
    # ---- 1. WHICH GAME -----------------------------------------------------
    # The card (side, rounds, K, fees, target) is read from the certified
    # environment and is never chosen in this file. 
    spec = Game().spec

    run_directory = results_path(f"torch_{spec.side}_R{spec.rounds}_{GATE}")
    run_directory.mkdir(parents=True, exist_ok=True)

    print(f"game (from rl_env): side {spec.side}, rounds {spec.rounds}, "
          f"K {spec.K}")
    print(f"  sample budget: {ITERS:,} iters x {BATCH:,} paths = "
          f"{ITERS * BATCH:,} training games")
    print("  method: direct policy optimisation through a differentiable "
          "simulator (neural stochastic control) -- NOT model-free RL")
    if GATE == "hard":
        print("  GATE=hard: autodiff through the "
              "indicator (known-biased gradient, kept to show it)")
    else:
        print(f"  GATE=smoothed: tau anneal {TAU_START} -> {TAU_END} x the "
              f"observed innovation scale, measured per batch (no "
              f"distribution constant anywhere; adapts to any sd)")

    if CHECKS:
        check_torch_game_matches_env(spec)
        check_policy_replay_matches_env(spec)

    start_time = time.time()

    trained_net, learning_curve = train(spec, run_directory)

    # A large fresh sample (never seen in training or validation), played on
    # the TRUE game -- tau=None, so real accept/reject, no smoothing. This is
    # a fast sanity number
    eval_generator = torch.Generator().manual_seed(SEED_PATH + 2)
    with torch.no_grad(): #Ensures only using network - not training it
        eval_pl = rollout(trained_net, spec,
                          torch.randn(EVAL_PATHS, spec.rounds + 1,
                                      generator=eval_generator),
                          tau=None)
    print(f"\nfinal (hard game, deterministic, {EVAL_PATHS:,} paths, torch "
          f"engine): P/L {float(eval_pl.mean()) * 100:+.4f}% "
          f"+/- {2 * float(eval_pl.std()) / math.sqrt(EVAL_PATHS) * 100:.4f}")

    # torch_policy() wraps the trained network as something the certified env
    # can drive. report() then replays it through THAT env -- solving the DP
    # fresh for comparison -- and writes the five standard outputs. 
    D.report(spec, torch_policy(trained_net, spec), run_directory,
             label="torch", curve=learning_curve)
    print(f"artefacts in {run_directory}/  "
          f"({(time.time() - start_time) / 60:.2f} min)")

if __name__ == "__main__":
    main()
