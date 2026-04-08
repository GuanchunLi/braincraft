# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3

"""
Deterministic Task 1 dummy2 player.

Variant of env1_player_dummy that keeps the same CW outer-ring trajectory
but controls steering during the wall-following phase using a heading-aligned
signal inferred from left/right depth symmetry, rather than a single
side-proximity ray. The side ray stays as a small safety valve.

Rationale: in the axis-aligned env1 world, when the bot's heading is a multiple
of 90 degrees the depth profile is symmetric around the camera center. A pair
of symmetric front rays (prox[L] - prox[R]) therefore acts as a linear proxy
for (heading mod 90 deg), and driving it to zero keeps the bot aligned with
the wall it is following. Corner turns and the first wall approach still rely
on the hit input for a hard CW snap.
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
    bias_idx = p + 2

    # Symmetric front pair around the camera midline (indices 0..63, center
    # between 31 and 32). L is left of center, R is its mirror.
    # Camera index 0 = leftmost ray, 63 = rightmost (see camera.py:137).
    L_idx = 20
    R_idx = 63 - L_idx  # = 43

    # Existing left-side ray, kept as a safety valve only.
    side_idx = 11

    # Feature neurons:
    #   X0 = relu_tanh(hit)                                     corner snap
    #   X1 = relu_tanh(bias) = tanh(1)                          constant
    #   X2 = relu_tanh(prox[L])                                 heading (+)
    #   X3 = relu_tanh(prox[R])                                 heading (-)
    #   X4 = relu_tanh(prox[side_idx])                          safety valve
    #   X5 = relu_tanh(prox[L] + prox[R] - front_thr)           front-block
    #        -> ~0 during cruising, >0 only when the forward field is
    #           obstructed. Dominates the heading term at corners so the
    #           heading signal cannot fight the CW turn when the right ray
    #           starts to see past the corner.
    Win[0, hit_idx] = 1.0
    Win[1, bias_idx] = 1.0
    Win[2, L_idx] = 1.0
    Win[3, R_idx] = 1.0
    Win[4, side_idx] = 1.0

    # Front-block gate must reflect the *actual* forward distance. Using the
    # off-center rays L_idx/R_idx gives false positives when the bot sits in
    # a narrow corridor (both rays pick up nearby side walls). Use the two
    # true center rays instead.
    C1_idx, C2_idx = 31, 32
    front_thr = 1.4  # cruise sum is ~1.3 at start, trips as front wall nears
    Win[5, C1_idx] = 1.0
    Win[5, C2_idx] = 1.0
    Win[5, bias_idx] = -front_thr

    TANH1 = np.tanh(1.0)  # ~0.7616 -- X1 value after relu_tanh

    # --- Gains (in RADIANS) ------------------------------------------------
    # Bot.forward interprets the readout as a dtheta in radians and clips it
    # to +/- 5 deg == +/- 0.0873 rad. Gains must live in that tiny band or
    # every term saturates every step.
    hit_turn = np.radians(-10.0) / TANH1   # saturates to -5 deg on hit
    # Heading term: want ~2-3 deg of correction per unit of (prox[L]-prox[R]).
    heading_gain = np.radians(8)
    # Front-block gate: strong right turn once activated; saturates easily.
    front_gain = np.radians(-20.0)
    # Safety valve: very small, only matters when the left side ray deviates
    # significantly from the target proximity.
    safety_gain = np.radians(-20.0)
    safety_target = 0.65
    # Safety term is   safety_gain*(X4 - tanh(safety_target))
    # and X1 carries the bias (value = TANH1), so the coefficient on X1 is
    # chosen to cancel at X4 = tanh(safety_target).
    safety_bias = -safety_gain * np.tanh(safety_target) / TANH1

    # O = hit_turn*X0
    #   + heading_gain*(X2 - X3)
    #   + front_gain*X5
    #   + safety_gain*X4 + safety_bias*X1
    Wout[0, 0] = hit_turn
    Wout[0, 2] = heading_gain
    Wout[0, 3] = -heading_gain
    Wout[0, 5] = front_gain
    Wout[0, 4] = safety_gain
    Wout[0, 1] = safety_bias

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
