# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3

"""
Deterministic Task 1 dummy3 player.

Same steering scaffold as env1_player_dummy2, with an internal reward-state
subcircuit and a post-reward shortcut detector inside X(t):

- X8 emits a transient pulse one step after the first observable energy rise,
- X9 latches `is_rewarded` from that pulse and stays near 1 thereafter,
- X10 stays silent until X9 is high, then detects the post-reward outermost-
  ray asymmetry that unlocks the shortcut turn.
"""

import time
import numpy as np

from bot import Bot
from environment_1 import Environment


def identity(x):
    return x


def relu_tanh(x):
    """Thresholded activation used for sparse hand-crafted feature neurons."""

    x = np.tanh(x)
    return np.where(x > 0, x, 0)


def dummy_player():
    """Build a deterministic outer-ring clockwise controller for Task 1
    whose wall-following is driven by heading-alignment symmetry."""

    bot = Bot()

    n = 1000
    p = bot.camera.resolution
    warmup = 0
    leak = 1.0
    # relu_tanh lets us implement a thresholded "front-blocked" gate via a
    # bias offset. It is still effectively linear on small non-negative
    # inputs (tanh(x) ~ x for small x), so the heading-symmetry term below
    # keeps behaving the way we want during straight wall following.
    f = relu_tanh
    g = identity

    Win = np.zeros((n, p + 3))
    W = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx = p
    energy_idx = p + 1
    bias_idx = p + 2

    # Symmetric front pair around the camera midline (indices 0..63, center
    # between 31 and 32). L is left of center, R is its mirror.
    # Camera index 0 = leftmost ray, 63 = rightmost (see camera.py:137).
    L_idx = 20
    R_idx = 63 - L_idx  # = 43

    # Existing left-side ray, kept as a safety valve only.
    left_side_idx = 11
    right_side_idx = 63 - left_side_idx

    # Feature neurons:
    #   X0 = relu_tanh(hit)                                     corner snap
    #   X1 = relu_tanh(prox[L])                                 heading (+)
    #   X2 = relu_tanh(prox[R])                                 heading (-)
    #   X3 = relu_tanh(prox[left_side_idx])                          safety valve (left side)
    #   X4 = relu_tanh(prox[right_side_idx])                          safety valve (right side)
    #   X5 = relu_tanh(prox[L] + prox[R] - front_thr)           front-block
    #        -> ~0 during cruising, >0 only when the forward field is
    #           obstructed. Dominates the heading term at corners so the
    #           heading signal cannot fight the CW turn when the right ray
    #           starts to see past the corner.
    Win[0, hit_idx] = 1.0
    Win[1, L_idx] = 1.0
    Win[2, R_idx] = 1.0
    Win[3, left_side_idx] = 1.0
    Win[4, right_side_idx] = 1.0

    # Front-block gate must reflect the *actual* forward distance. Using the
    # off-center rays L_idx/R_idx gives false positives when the bot sits in
    # a narrow corridor (both rays pick up nearby side walls). Use the two
    # true center rays instead.
    C1_idx, C2_idx = 31, 32
    front_thr = 1.4  # cruise sum is ~1.3 at start, trips as front wall nears
    Win[5, C1_idx] = 1.0
    Win[5, C2_idx] = 1.0
    Win[5, bias_idx] = -front_thr

    # ------------------------------------------------------------------
    # Reward-state internal circuit.
    #
    #   X6  = delayed copy of scaled energy
    #   X7  = "armed" latch, used only to suppress the startup false positive
    #   X8  = transient reward pulse after the first net energy increase
    #   X9  = latched is_rewarded: 0 until first refill, then pinned at ~1
    #   X10 = post-reward outermost-ray asymmetry detector
    #
    # Update rule is X(t+1) = relu_tanh(Win @ I(t) + W @ X(t)) with leak=1.
    # ------------------------------------------------------------------
    K = 0.005
    arm_from_energy = 1000.0
    arm_latch = 10.0
    pulse_gain = 100000.0
    pulse_thr = 0.2
    arm_gate = 1000.0
    latch_gain = 10.0

    # X6 tracks the previous step's energy in a near-linear regime so the
    # pulse detector can compare consecutive samples without the startup
    # sample looking like a reward.
    Win[6, energy_idx] = K

    # X7 becomes high once X6 has stored one valid energy sample and then
    # stays high. This gates X8 off during the uninitialized startup step.
    W[7, 6] = arm_from_energy
    W[7, 7] = arm_latch

    # X8: pulse on the first positive net energy jump after the arm is high.
    # The large positive gate from X7 is cancelled by the bias until armed.
    Win[8, energy_idx] = pulse_gain * K
    W[8, 6] = -pulse_gain
    W[8, 7] = arm_gate
    Win[8, bias_idx] = -(arm_gate + pulse_thr)

    # ------------------------------------------------------------------
    # X9: self-exciting is_rewarded latch driven by the transient X8 pulse.
    # Once set, it stays pinned near 1 for the remainder of the run.
    # ------------------------------------------------------------------
    W[9, 8] = latch_gain
    W[9, 9] = latch_gain

    # ------------------------------------------------------------------
    # X10: post-reward outermost-ray asymmetry detector.
    # Silent (clamped to 0) while X9 ~ 0, so behavior before the first
    # reward is identical to dummy2 / V1 dummy3. Once X9 latches to ~1,
    # fires when prox[63] exceeds prox[0] by `margin`,
    # driving an extra right turn that lets the bot peel off into the
    # inner corridor and run only half of the outer ring.
    # ------------------------------------------------------------------
    K9 = 3.0
    margin = 0.0
    G9 = 10.0
    Win[10, 0] = -K9
    Win[10, 63] = K9
    Win[10, bias_idx] = -(G9 + K9 * margin)
    W[10, 9] = G9

    TANH1 = np.tanh(1.0)  # ~0.7616 -- X1 value after relu_tanh

    # --- Gains (in RADIANS) ------------------------------------------------
    # Bot.forward interprets the readout as a dtheta in radians and clips it
    # to +/- 5 deg == +/- 0.0873 rad. Gains must live in that tiny band or
    # every term saturates every step.
    hit_turn = np.radians(-10.0) / TANH1   # saturates to -5 deg on hit
    # Heading term: want ~2-3 deg of correction per unit of (prox[L]-prox[R]).
    heading_gain = np.radians(0)
    # Safety valve: very small, only matters when the left side ray deviates
    # significantly from the target proximity.
    # Safety term is safety_gain_left*relu_tanh(X3 - safety_target) + safety_gain_right*relu_tanh(X4 - safety_target)
    safety_gain_left = np.radians(-20.0)
    safety_gain_right= -safety_gain_left
    safety_target = 0.7
    Win[3, bias_idx] = -safety_target
    Win[4, bias_idx] = -safety_target
    # Front-block gate: strong right turn once activated; saturates easily.
    front_gain = np.radians(-20.0)
    # Post-reward shortcut: extra right turn once X9 has latched and the
    # right side opens up while the left wall is still present.
    post_reward_gain = 0.0 * front_gain # No steering yet
    # heading_gain += post_reward_gain

    # O = hit_turn*X0
    #   + heading_gain*(X1 - X2)
    #   + safety_term
    #   + front_gain*X5
    #   + post_reward_gain*X10
    Wout[0, 0] = hit_turn
    Wout[0, 1] = heading_gain
    Wout[0, 2] = -heading_gain
    Wout[0, 3] = safety_gain_left
    Wout[0, 4] = safety_gain_right
    Wout[0, 5] = front_gain
    Wout[0, 10] = post_reward_gain

    model = Win, W, Wout, warmup, leak, f, g
    yield model


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from challenge_1 import train, evaluate

    seed = 12345

    np.random.seed(seed)
    print("Starting training for 100 seconds (user time)")
    model = train(dummy_player, timeout=100)

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed, runs=4)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
