# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3

"""
Deterministic Task 1 dummy player.

The controller is intentionally simple: it drives into the outer ring, turns
clockwise on front-wall contact, and keeps following that ring with a small set
of hand-wired feature neurons.
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
    """Build a deterministic outer-ring clockwise controller for Task 1."""

    bot = Bot()

    n = 1000
    p = bot.camera.resolution
    warmup = 0
    leak = 1.0
    f = identity
    g = identity

    Win = np.zeros((n, p + 3))
    W = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx = p
    bias_idx = p + 2

    # Left-side wall follower (mirrors env3_player_wallfollow.py).
    # Camera index 0 = leftmost ray, 63 = rightmost (see camera.py:137).
    # Use a single ray inside the left half (not the extreme ray) for stable
    # wall-following: side proximity = 1 - depth[side_idx].
    side_idx = 11

    # Internal state: X0=hit, X1=bias, X2=side proximity.
    Win[0, hit_idx] = 1.0
    Win[1, bias_idx] = 1.0
    Win[2, side_idx] = 1.0

    hit_turn = -5.0          # negative -> hard right (CW) on contact
    wall_gain = -0.40        # negative so left-closer steers right
    wall_target = 0.65       # target proximity (~ depth 0.35)

    # O = hit_turn*hit + wall_gain*(side - wall_target)
    Wout[0, 0] = hit_turn
    Wout[0, 1] = -wall_gain * wall_target
    Wout[0, 2] = wall_gain

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
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
