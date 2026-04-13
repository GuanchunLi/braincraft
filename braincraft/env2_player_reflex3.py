# Braincraft challenge - Reflex Player v3 adapted for Environment 2
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3

"""
Reflex Player v3 for Environment 2.

This keeps the reflex2 wall-following, initial heading correction, and
corridor shortcut behavior intact. The only behavioral change is how the
front-block steering gain is signed:

* reflex2: fixed front_gain = -np.radians(20.0)
* reflex3: front_gain sign is latched from early color evidence

Because the bot starts with a small random heading jitter, the blue wall is
not always visible on the very first camera frame. Reflex3 therefore gathers
evidence during the initial few activations and latches the sign as soon as
the side choice is unambiguous. Before that latch, the front-block term is
held at zero.
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_2 import Environment
import env2_player_reflex2 as reflex2


identity = reflex2.identity
relu_tanh = reflex2.relu_tanh


COLOR_COPY_START = 25
BLUE_WALL_VALUE = 4.0
COLOR_EVIDENCE_THRESHOLD = 2.0
FRONT_GAIN_MAG = np.radians(20.0)
DEFAULT_SPEED = 0.01


def _reflex3_indices(n_rays):
    color_copy_stop = COLOR_COPY_START + n_rays
    left_blue_idx = color_copy_stop
    right_blue_idx = left_blue_idx + 1
    evidence_idx = right_blue_idx + 1
    front_sign_idx = evidence_idx + 1
    signed_front_idx = front_sign_idx + 1
    return (
        color_copy_stop,
        left_blue_idx,
        right_blue_idx,
        evidence_idx,
        front_sign_idx,
        signed_front_idx,
    )


def make_activation(
    speed,
    wout_feature_weights,
    front_gain_mag,
    n_rays,
    wout_shortcut_weight=0.0,
):
    """Custom activation for reflex3.

    This is the reflex2 activation with one targeted change: the front-block
    contribution is routed through a latched sign chosen from early color
    evidence instead of a fixed sign in Wout[0, 5].
    """

    a = np.radians(5)
    wout = np.asarray(wout_feature_weights, dtype=float)
    _ = float(wout_shortcut_weight)

    (
        color_copy_stop,
        left_blue_idx,
        right_blue_idx,
        evidence_idx,
        front_sign_idx,
        signed_front_idx,
    ) = _reflex3_indices(n_rays)

    def f(x):
        if x.ndim == 1:
            is2d = False
        elif x.ndim == 2 and x.shape[1] == 1:
            is2d = True
        else:
            raise ValueError(
                "make_activation expects shape (n,) or (n, 1); "
                f"got {x.shape}"
            )

        def _g(arr, i):
            return float(arr[i, 0]) if is2d else float(arr[i])

        def _s(arr, i, v):
            if is2d:
                arr[i, 0] = v
            else:
                arr[i] = v

        out = np.empty_like(x)

        # X0-X5: same reflex features as reflex2.
        out[:6] = relu_tanh(x[:6])

        # X12: hit signal.
        out[12:13] = relu_tanh(x[12:13])

        # X15-X18: reward circuit.
        out[15:19] = relu_tanh(x[15:19])

        # Early color evidence for the front-block sign latch.
        if is2d:
            raw_colors = x[COLOR_COPY_START:color_copy_stop, 0]
        else:
            raw_colors = x[COLOR_COPY_START:color_copy_stop]

        half = n_rays // 2
        left_blue = bool(np.any(np.isclose(raw_colors[:half], BLUE_WALL_VALUE)))
        right_blue = bool(np.any(np.isclose(raw_colors[half:], BLUE_WALL_VALUE)))

        left_evidence = 1.0 if left_blue and not right_blue else 0.0
        right_evidence = 1.0 if right_blue and not left_blue else 0.0
        _s(out, left_blue_idx, left_evidence)
        _s(out, right_blue_idx, right_evidence)

        evidence_prev = _g(x, evidence_idx)
        front_sign_prev = _g(x, front_sign_idx)
        if abs(front_sign_prev) < 0.5:
            evidence_now = evidence_prev + right_evidence - left_evidence
            if evidence_now >= COLOR_EVIDENCE_THRESHOLD:
                front_sign = 1.0
            elif evidence_now <= -COLOR_EVIDENCE_THRESHOLD:
                front_sign = -1.0
            else:
                front_sign = 0.0
        else:
            evidence_now = evidence_prev
            front_sign = front_sign_prev

        _s(out, evidence_idx, evidence_now)
        _s(out, front_sign_idx, front_sign)
        _s(out, signed_front_idx, front_sign * _g(out, 5))

        # Same lagged heading state as reflex2.
        val7 = _g(x, 7)
        x13 = _g(x, 13)
        x14 = _g(x, 14)

        if abs(val7) < 1e-8:
            correction = x13
            hold_val = x13
        else:
            correction = x13 - x14
            hold_val = correction

        theta_lagged = val7 + np.pi / 2 + correction
        sin_lagged = np.sin(theta_lagged)

        # Initial heading correction X23/X24 from reflex2.
        x23_prev = _g(x, 23)
        x24_prev = _g(x, 24)

        if x24_prev < 0.5:
            init_remaining = float(
                np.clip(-correction, -reflex2.INIT_CORR_CAP, reflex2.INIT_CORR_CAP)
            )
            init_dtheta = 0.0
            init_correction_active = False
            x23_new = init_remaining
        else:
            init_remaining = x23_prev
            if abs(init_remaining) > reflex2.INIT_CORR_EPS:
                init_dtheta = float(np.clip(init_remaining, -a, a))
                init_correction_active = True
                x23_new = init_remaining - init_dtheta
            else:
                init_dtheta = 0.0
                init_correction_active = False
                x23_new = 0.0

        _s(out, 23, x23_new)
        _s(out, 24, 1.0)

        # Step counter X19.
        x10_prev = _g(x, 10)
        x19_val = _g(x, 19)
        near_center = abs(x10_prev) < reflex2.NEAR_CENTER_THR
        heading_vert = abs(sin_lagged) > reflex2.SIN_VERT_THR

        if near_center and heading_vert:
            counter = 0.0
        else:
            counter = x19_val + 1.0
        _s(out, 19, counter)

        # Shortcut steering X20.
        is_rewarded = _g(out, 18) > 0.5
        heading_horiz = abs(sin_lagged) < reflex2.SIN_HORIZ_THR
        front_clear = _g(out, 5) < 0.1
        enough_steps = x19_val > reflex2.COUNTER_THR
        cos_lagged = np.cos(theta_lagged)
        if abs(cos_lagged) > 0.5:
            trig_offset = -reflex2.DRIFT_OFFSET * np.sign(cos_lagged)
        else:
            trig_offset = 0.0
        near_corr = abs(x10_prev - trig_offset) < reflex2.NEAR_CENTER_THR

        x22_prev = _g(x, 22) if x.shape[0] > 22 else 0.0
        trigger = (
            is_rewarded
            and heading_horiz
            and front_clear
            and enough_steps
            and near_corr
            and x22_prev < 0.5
        )

        if trigger:
            sc_countdown = float(reflex2.SC_TOTAL)
        elif x22_prev > 0.5:
            sc_countdown = x22_prev - 1.0
        else:
            sc_countdown = 0.0

        is_turning = sc_countdown > float(reflex2.APPROACH_STEPS) + 0.5
        is_approaching = sc_countdown > 0.5 and not is_turning

        x11_prev = _g(x, 11)
        if abs(x11_prev) > 0.01:
            turn_toward = -np.sign(cos_lagged) * np.sign(x11_prev)
        else:
            turn_toward = -1.0

        if is2d:
            O_main = float(wout @ out[:6, 0])
            O_no_front = float(wout[:5] @ out[:5, 0])
        else:
            O_main = float(wout @ out[:6])
            O_no_front = float(wout[:5] @ out[:5])
        O_features = O_main + front_gain_mag * _g(out, signed_front_idx)

        _s(out, 22, sc_countdown)

        if init_correction_active:
            x20_val = init_dtheta - O_features
        elif is_turning:
            x20_val = abs(reflex2.SHORTCUT_TURN) * turn_toward
        elif is_approaching:
            x20_val = -O_no_front
        else:
            x20_val = 0.0
        _s(out, 20, x20_val)

        O_now = O_features + x20_val
        dtheta_now = np.clip(O_now, -a, a)
        _s(out, 6, dtheta_now)

        out[7:8] = x[7:8]

        theta_now = val7 + np.pi / 2 + correction + dtheta_now
        cos_now = np.cos(theta_now)
        sin_now = np.sin(theta_now)
        _s(out, 8, cos_now)
        _s(out, 9, sin_now)

        hit_val = _g(x, 12)
        if hit_val < 0.5:
            _s(out, 10, _g(x, 10) + speed * cos_now)
            _s(out, 11, _g(x, 11) + speed * sin_now)
        else:
            _s(out, 10, _g(x, 10))
            _s(out, 11, _g(x, 11))

        _s(out, 13, hold_val)
        _s(out, 14, x14)

        # Preserve raw color copies for this activation; no recurrence needed.
        out[COLOR_COPY_START:color_copy_stop] = x[COLOR_COPY_START:color_copy_stop]

        if x.shape[0] > signed_front_idx + 1:
            out[signed_front_idx + 1 :] = relu_tanh(x[signed_front_idx + 1 :])

        return out

    return f


def reflex3_player():
    """Build reflex player v3 for env2.

    The network is reflex2 plus a small early-evidence latch that chooses the
    sign of the front-block steering term and keeps it fixed for the run.
    """

    Win, W, Wout, warmup, leak, _f, g = next(reflex2.reflex2_player())
    Win = Win.copy()
    W = W.copy()
    Wout = Wout.copy()

    n_rays = (Win.shape[1] - 3) // 2
    speed = DEFAULT_SPEED

    (
        color_copy_stop,
        _left_blue_idx,
        _right_blue_idx,
        evidence_idx,
        front_sign_idx,
        signed_front_idx,
    ) = _reflex3_indices(n_rays)

    # Copy the raw color wall ids into unused neurons so the activation can
    # read the exact early observation directly.
    for ray_idx in range(n_rays):
        Win[COLOR_COPY_START + ray_idx, n_rays + ray_idx] = 1.0

    # Evidence accumulator and sign latch.
    W[evidence_idx, evidence_idx] = 1.0
    W[front_sign_idx, front_sign_idx] = 1.0

    # Remove the fixed front_gain on X5 and replace it with a signed copy.
    Wout[0, 5] = 0.0
    Wout[0, signed_front_idx] = FRONT_GAIN_MAG

    wout_features = Wout[0, :6].copy()
    f = make_activation(speed, wout_features, FRONT_GAIN_MAG, n_rays)

    model = Win, W, Wout, warmup, leak, f, g
    yield model


if __name__ == "__main__":
    import time
    from challenge_2 import evaluate, train

    seed = 12345
    np.random.seed(seed)
    print("Training reflex3 player for env2 (single yield, instant)...")
    model = train(reflex3_player, timeout=100)

    W_in, W, W_out, warmup, leak, f, g = model

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score (distance): {score:.2f} +/- {std:.2f}")
