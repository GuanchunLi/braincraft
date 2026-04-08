# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3

"""
Deterministic Task 1 dummy3 player.

Same steering controller as env1_player_dummy2 (identical output), with an
additional internal reservoir state `is_rewarded` latched inside X(t):

- X8 starts at 0,
- flips to ~1 the first step after the bot's energy is replenished by an
  energy source (i.e. energy(t) > energy(t-1)),
- stays at ~1 for the remainder of the run.

This is implemented with three extra neurons (X6: delayed energy copy,
X7: reward pulse detector, X8: self-exciting latch) and zero contribution to
Wout, so the steering behavior is identical to dummy2.
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
    # is_rewarded internal state (does NOT contribute to output).
    #
    #   X6 = delayed copy of (scaled) energy
    #   X7 = reward pulse: saturates to ~1 on the step where energy jumps up
    #   X8 = latched is_rewarded: 0 until first refill, then pinned at ~1
    #
    # Update rule is X(t+1) = relu_tanh(Win @ I(t) + W @ X(t)) with leak=1.
    # ------------------------------------------------------------------
    K = 0.1        # near-linear scale for energy copy (tanh(0.1) ~ 0.0997)
    K_big = 100.0  # high-gain edge detector
    latch_gain = 10.0

    # X6: X6(t+1) = relu_tanh(K * energy(t))  ~  K * energy(t)
    Win[6, energy_idx] = K

    # X7: relu_tanh( K_big * K * energy(t) - K_big * X6(t) )
    #   ~ relu_tanh( K_big * K * (energy(t) - energy(t-1)) )
    # Normal per-step drain (~1e-3) gives ~ -0.01 -> clamped to 0.
    # A refill jump gives a large positive argument -> saturates near +1.
    Win[7, energy_idx] = K_big * K
    W[7, 6] = -K_big

    # X8: self-exciting latch set by the reward pulse.
    # X8(t+1) = relu_tanh(latch_gain * X8(t) + latch_gain * X7(t))
    # Once either input is appreciably positive, argument >> 1 -> X8 ~ 1,
    # and the self-feedback then keeps it pinned near 1 for the rest of the
    # run (no decay, since leak = 1.0).
    W[8, 7] = latch_gain
    W[8, 8] = latch_gain
    # The is_rewarded internal state lives at X[8].

    # ------------------------------------------------------------------
    # X9: post-reward "left wall present, right side open" detector.
    # Silent (clamped to 0) while X8 ~ 0, so behavior before the first
    # reward is identical to dummy2 / V1 dummy3. Once X8 latches to ~1,
    # fires when prox[left_side] exceeds prox[right_side] by `margin`,
    # driving an extra right turn that lets the bot peel off into the
    # inner corridor and run only half of the outer ring.
    # ------------------------------------------------------------------
    K9 = 2.0
    margin = 0.3
    G8 = 10.0
    Win[9, left_side_idx]  =  K9
    Win[9, right_side_idx] = -K9
    Win[9, bias_idx]       = -(G8 + K9 * margin)
    W[9, 8] = G8

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
    # Post-reward shortcut: extra right turn once X8 has latched and the
    # right side opens up while the left wall is still present.
    post_reward_gain = 5.0 * front_gain
    # heading_gain += post_reward_gain

    # O = hit_turn*X0
    #   + heading_gain*(X1 - X2)
    #   + front_gain*X3
    #   + safety_term
    Wout[0, 0] = hit_turn
    Wout[0, 1] = heading_gain
    Wout[0, 2] = -heading_gain
    Wout[0, 3] = safety_gain_left
    Wout[0, 4] = safety_gain_right
    Wout[0, 5] = front_gain
    Wout[0, 9] = post_reward_gain

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
