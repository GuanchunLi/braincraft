# Braincraft challenge — Bio Player for Environment 3
# Copyright (C) 2026 Guanchun Li
# Released under the GNU General Public License 3

"""
Bio Player for Environment 3.

A pointwise-activation Echo State Network controller: every hidden
activation f(x)[i] depends only on x[i], and all cross-neuron logic is
carried by the connectivity matrices. The update is

    X(t+1) = f(Win @ I(t) + W @ X(t))
    O(t+1) = Wout @ g(X(t+1))

with identity readout g(x) = x.

Env3 exposes colour (the sources are coloured), but this controller does
not read colour: the bot simply runs around the outer corridor with a
reflex wall-follower and picks up whatever source it passes over. The
env1 shortcut and reward circuits are removed, together with every
neuron that existed only to support them (heading/position integrators,
trig helpers, corridor predicates, quadrant ANDs, etc.).

The input vector keeps the full env3/env2 layout,

    I(t) = [prox[0..63](t), colour[0..63](t), hit(t), energy(t), 1]     (2p + 3 = 131 cols)

because env3's challenge runner builds it that way; only the proximity,
hit and bias columns receive any input weight.

The hidden pool packs one functional slot per neuron (12 slots total,
slots 12..999 are unused):

    0..4    reflex features (hit, proximity, safety)
    5       dtheta (clipped one-step-lagged steering command)
    6..9    initial-heading correction timing and seeds
    10      initial-impulse steering actuator
    11      unsigned front-block escape channel

Constants use snake_case; local hidden-state aliases use UPPER_SNAKE.
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_3 import Environment


# ── Bio-specific constants ────────────────────────────────────────────
front_gain_mag     = np.radians(20.0)
k_sharp            = 50.0            # logistic-like gain for threshold gates
step_a             = np.radians(5.0) # actuator clip (±5°)
seed_window_k      = 6               # initial-heading correction window length


# ── Neuron index layout ───────────────────────────────────────────────
def _bio_indices():
    """Sequential neuron-slot map for the env3 bio controller."""
    idx = {
        "hit_feat":      0,
        "prox_left":     1,
        "prox_right":    2,
        "safe_left":     3,
        "safe_right":    4,
        "dtheta":        5,
        "step_counter":  6,
        "seeded_flag":   7,
        "seed_pos":      8,
        "seed_neg":      9,
        "init_impulse": 10,
        "front_block":  11,
    }
    idx["bio_end"] = 12
    return idx


# ── Pointwise activation dispatch ─────────────────────────────────────
def make_activation(a, idx):
    """Build the per-neuron pointwise activation.

    Each out[i] depends only on x[i]. The activation used for neuron i is
    picked once here from precomputed index arrays:

        identity  : step_counter, init_impulse
        clip_a    : dtheta, clipped to ±step_a
        default   : max(0, tanh(z)) — reflex channels, latch, seed gates,
                    front-block detector
    """
    id_list = [idx["step_counter"], idx["init_impulse"]]
    id_arr  = np.array(sorted(set(id_list)), dtype=int)

    def f(x):
        out = np.maximum(0.0, np.tanh(x))  # default relu_tanh
        if id_arr.size:
            out[id_arr, 0] = x[id_arr, 0]
        out[idx["dtheta"], 0] = float(np.clip(x[idx["dtheta"], 0], -a, a))
        return out

    return f


# ── Player builder ────────────────────────────────────────────────────
def bio_player():
    """Build the bio controller for env3. Yields a single frozen model."""

    bot = Bot()
    n = 1000
    p = bot.camera.resolution          # 64
    warmup = 0
    leak = 1.0
    g = lambda x: x                    # identity readout

    # Env3 feeds I = [depths, colours, hit, energy, 1]. We do not read
    # colour, but must index hit/energy/bias at the full 2p+3 offsets.
    n_inputs = 2 * p + 3               # 131
    Win  = np.zeros((n, n_inputs))
    W    = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx    = 2 * p                 # 128
    bias_idx   = 2 * p + 2             # 130

    idx = _bio_indices()
    a = step_a

    # Local aliases — keep wiring lines short and readable.
    HIT_FEAT     = idx["hit_feat"]
    PROX_LEFT    = idx["prox_left"]
    PROX_RIGHT   = idx["prox_right"]
    SAFE_LEFT    = idx["safe_left"]
    SAFE_RIGHT   = idx["safe_right"]
    DTHETA       = idx["dtheta"]
    STEP_COUNTER = idx["step_counter"]
    SEEDED_FLAG  = idx["seeded_flag"]
    SEEDP        = idx["seed_pos"]
    SEEDN        = idx["seed_neg"]
    INIT_IMPULSE = idx["init_impulse"]
    FB           = idx["front_block"]

    # Ray indices sampled by the reflex/front channels.
    L_idx          = 20
    R_idx          = 63 - L_idx         # 43
    left_side_idx  = 11
    right_side_idx = 63 - left_side_idx # 52
    C1_idx, C2_idx = 31, 32              # two centre-front proximity taps
    front_thr      = 1.4

    TANH1 = np.tanh(1.0)

    hit_turn          = np.radians(-10.0) / TANH1
    heading_gain      = np.radians(-40.0)
    safety_gain_left  = np.radians(-20.0)
    safety_gain_right = -safety_gain_left
    safety_target     = 0.75

    # ── Reflex features and steering readout ──────────────────────────
    Win[HIT_FEAT,    hit_idx]        = 1.0
    Win[PROX_LEFT,   L_idx]          = 1.0
    Win[PROX_RIGHT,  R_idx]          = 1.0
    Win[SAFE_LEFT,   left_side_idx]  = -1.0
    Win[SAFE_RIGHT,  right_side_idx] = -1.0
    Win[SAFE_LEFT,   bias_idx]       = safety_target
    Win[SAFE_RIGHT,  bias_idx]       = safety_target

    Wout[0, HIT_FEAT]     = hit_turn
    Wout[0, PROX_LEFT]    = heading_gain
    Wout[0, PROX_RIGHT]   = -heading_gain
    Wout[0, SAFE_LEFT]    = safety_gain_left
    Wout[0, SAFE_RIGHT]   = safety_gain_right
    Wout[0, INIT_IMPULSE] = 1.0
    # Wout[0, FB] is set below after the front-block wiring.

    # ── Initial-heading correction latch ──────────────────────────────
    # step_counter is a plain identity counter (= t). seeded_flag is a
    # sharp threshold against it: 0 for t = 0..seed_window_k - 1, then
    # saturates to 1. seed_pos / seed_neg read seeded_flag(t), so they
    # fire for exactly seed_window_k consecutive steps (t = 1..K).
    # init_impulse adds -(seed_pos - seed_neg) = -(R - L)/0.173 to the
    # steering output each of those steps, closing a proportional loop
    # that drives the residual left/right depth asymmetry toward zero.
    cal_gain = 1.0 / 0.173

    W[STEP_COUNTER, STEP_COUNTER] = 1.0
    Win[STEP_COUNTER, bias_idx]   = 1.0

    W[SEEDED_FLAG, STEP_COUNTER]  = k_sharp
    Win[SEEDED_FLAG, bias_idx]    = -k_sharp * (float(seed_window_k) - 1.5)

    # SEED_POS/NEG: signed (R - L) depth asymmetry, gated off once
    # seeded_flag saturates (-1000 gate drives the pre-activation well
    # below zero so relu_tanh outputs zero).
    Win[SEEDP, L_idx]     = -cal_gain
    Win[SEEDP, R_idx]     =  cal_gain
    W[SEEDP, SEEDED_FLAG] = -1.0e3
    Win[SEEDN, L_idx]     =  cal_gain
    Win[SEEDN, R_idx]     = -cal_gain
    W[SEEDN, SEEDED_FLAG] = -1.0e3

    # init_impulse = -(seed_pos - seed_neg): identity-activation actuator
    # that feeds straight into Wout during the seed window.
    W[INIT_IMPULSE, SEEDP] = -1.0
    W[INIT_IMPULSE, SEEDN] =  1.0

    # ── Front-block channel (unsigned, no colour gating) ──────────────
    # Fires when the two centre proximity taps exceed front_thr. With no
    # colour cue used, a single unsigned channel drives a fixed-direction
    # escape turn via Wout[0, FB].
    Win[FB, C1_idx]   = 1.0
    Win[FB, C2_idx]   = 1.0
    Win[FB, bias_idx] = -front_thr

    Wout[0, FB] = front_gain_mag

    # ── dtheta readout tie-back ───────────────────────────────────────
    # z_dtheta(t+1) = Wout @ X(t) = O(t), implemented by mirroring the
    # output row into W[DTHETA, :]. After clipping, dtheta(t+1) is the
    # previous step's steering command bounded to ±step_a. It is not
    # read by any other neuron but is kept so the readout is observable
    # as a hidden-state variable.
    for j in range(n):
        if Wout[0, j] != 0.0:
            W[DTHETA, j] = Wout[0, j]

    f = make_activation(a, idx)
    model = Win, W, Wout, warmup, leak, f, g
    yield model


if __name__ == "__main__":
    import time
    from challenge_3 import evaluate, train

    seed = 12345
    np.random.seed(seed)
    print("Training bio player for env3...")
    model = train(bio_player, timeout=100)

    W_in, W, W_out, warmup, leak, f, g = model

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score (distance): {score:.2f} +/- {std:.2f}")
