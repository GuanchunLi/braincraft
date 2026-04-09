# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3

"""
Deterministic Task 1 reflex player.

Combines dummy4's steering controller and internal state tracking with
dummy3's reward-state detection circuit.  Output is identical to dummy4
(and dummy2) — the reward-state neurons do not contribute to steering yet.

Internal state neurons (all with zero contribution to Wout):

- X6:  current clamped dtheta (computed in activation from features)
       Stored for X7 to accumulate on the next step.

- X7:  direction accumulator (activation: identity)
       Cumulative dtheta with 1-step lag.  The activation function adds
       the current step's dtheta on the fly for zero-lag direction.

- X8:  cos(direction_model) -- zero-lag, computed in activation.
- X9:  sin(direction_model) -- zero-lag, computed in activation.

- X10: x-displacement accumulator (gated by hit signal)
       model_x = 0.5 + X10

- X11: y-displacement accumulator (gated by hit signal)
       model_y = 0.5 + X11

- X12: hit signal from input (relu_tanh)

- X13: latched initial-heading correction (sample-and-hold)
       At step 0 the sensor asymmetry (prox[R]-prox[L]) is used to
       estimate the unknown +/-5 deg initial-direction noise.  The
       estimate is latched via self-recurrence and subtracted from the
       current-step sensor contribution using neuron 14.

- X14: instantaneous sensor correction (no recurrence)
       Same Win weights as X13 but no self-recurrence, so it always
       reflects the CURRENT sensor asymmetry.  X13 - X14 recovers the
       latched first-step value at all subsequent steps.

Reward-state circuit (remapped from dummy3 X6-X9):

- X15: delayed copy of scaled energy
- X16: "armed" latch, suppresses the startup false positive
- X17: reward detector that stays high while energy is increasing and the
       circuit is armed
- X18: latched is_rewarded: 0 until first refill, then pinned at ~1

The steering behavior is identical to dummy2 because Wout[0, 6:] == 0.
"""

import time
import numpy as np
if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_1 import Environment


def identity(x):
    return x


def relu_tanh(x):
    """Thresholded activation used for sparse hand-crafted feature neurons."""
    x = np.tanh(x)
    return np.where(x > 0, x, 0)


def make_activation(speed, wout_feature_weights):
    """Build a custom per-neuron activation with zero-lag direction tracking
    and first-step heading calibration.

    The activation function:
    1. Re-computes the current output O from the freshly-activated feature
       neurons, clips it, and adds it to the 1-step-lagged accumulator
       for zero-lag direction.
    2. At step 0 (detected by X7 == 0), latches a heading correction
       derived from the L/R sensor asymmetry (sample-and-hold).
    3. Integrates position using cos/sin of the corrected, zero-lag
       direction, freezing on hit steps.

    Parameters
    ----------
    speed : float
        Bot speed (0.01), used for position dead reckoning.
    wout_feature_weights : array, shape (6,)
        Wout[0, 0:6] -- output weights for the 6 feature neurons.
    """
    a = np.radians(5)  # +/-5 degrees clamp limit
    wout = np.asarray(wout_feature_weights, dtype=float)

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

        out = np.empty_like(x)

        # -- Feature neurons (relu_tanh, same as dummy2) --
        out[:6] = relu_tanh(x[:6])

        # -- Compute current dtheta from just-activated features --
        if is2d:
            O_now = float(wout @ out[:6, 0])
        else:
            O_now = float(wout @ out[:6])
        dtheta_now = np.clip(O_now, -a, a)

        # -- Store current dtheta in X6 (zero-lag, not delayed) --
        if is2d:
            out[6, 0] = dtheta_now
        else:
            out[6] = dtheta_now

        # -- Direction accumulator (identity) --
        # x[7] = X_old[7] + X_old[6] = cumulative dtheta (steps 0..t-1)
        out[7:8] = x[7:8]

        # -- Sample-and-hold heading correction --
        # x[13] = X_old[13] + cal*(prox_R - prox_L)   (with self-recurrence)
        # x[14] =              cal*(prox_R - prox_L)   (no recurrence)
        val7 = float(x[7, 0]) if is2d else float(x[7])
        x13  = float(x[13, 0]) if is2d else float(x[13])
        x14  = float(x[14, 0]) if is2d else float(x[14])

        if abs(val7) < 1e-8:
            # Step 0: sample correction from sensor asymmetry
            correction = x13          # = cal * (prox_R_0 - prox_L_0)
            hold_val   = x13          # latch this value
        else:
            # Step 1+: hold -- subtract current sensor contribution
            correction = x13 - x14   # = X_old[13] = latched correction
            hold_val   = correction   # keep latched value

        # -- Zero-lag direction with correction --
        theta_now = val7 + np.pi / 2 + correction + dtheta_now
        cos_now = np.cos(theta_now)
        sin_now = np.sin(theta_now)

        # -- Store cos / sin of current direction --
        if is2d:
            out[8, 0] = cos_now
            out[9, 0] = sin_now
        else:
            out[8] = cos_now
            out[9] = sin_now

        # -- Position accumulators (gated by hit) --
        hit_val = float(x[12, 0]) if is2d else float(x[12])
        if is2d:
            if hit_val < 0.5:
                out[10, 0] = x[10, 0] + speed * cos_now
                out[11, 0] = x[11, 0] + speed * sin_now
            else:
                out[10, 0] = x[10, 0]
                out[11, 0] = x[11, 0]
        else:
            if hit_val < 0.5:
                out[10] = x[10] + speed * cos_now
                out[11] = x[11] + speed * sin_now
            else:
                out[10] = x[10]
                out[11] = x[11]

        # -- Hit signal --
        out[12:13] = relu_tanh(x[12:13])

        # -- Correction neurons (identity / sample-and-hold) --
        if is2d:
            out[13, 0] = hold_val
            out[14, 0] = x14       # identity pass-through
        else:
            out[13] = hold_val
            out[14] = x14

        # -- Remaining neurons (includes reward circuit X15-X18) --
        if x.shape[0] > 15:
            out[15:] = relu_tanh(x[15:])

        return out

    return f


def reflex_player():
    """Build a deterministic controller identical to dummy2 output,
    with internal head direction, position tracking, and reward-state
    detection."""

    bot = Bot()

    n = 1000
    p = bot.camera.resolution   # 64
    warmup = 0
    leak = 1.0
    g = identity

    Win  = np.zeros((n, p + 3))
    W    = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx    = p          # 64
    energy_idx = p + 1      # 65
    bias_idx   = p + 2      # 66

    speed = bot.speed       # 0.01

    # ===== Feature neurons (identical to dummy2) =============================
    L_idx          = 20
    R_idx          = 63 - L_idx          # 43
    left_side_idx  = 11
    right_side_idx = 63 - left_side_idx  # 52

    Win[0, hit_idx]        = 1.0           # X0: hit
    Win[1, L_idx]          = 1.0           # X1: prox[L]
    Win[2, R_idx]          = 1.0           # X2: prox[R]
    Win[3, left_side_idx]  = -1.0          # X3: safety L
    Win[4, right_side_idx] = -1.0          # X4: safety R

    C1_idx, C2_idx = 31, 32
    front_thr      = 1.4
    Win[5, C1_idx]   = 1.0
    Win[5, C2_idx]   = 1.0
    Win[5, bias_idx] = -front_thr          # X5: front-block

    TANH1 = np.tanh(1.0)

    # ===== Gains (identical to dummy2) =======================================
    hit_turn          = np.radians(-10.0) / TANH1
    heading_gain      = np.radians(-40)
    safety_gain_left  = np.radians(-20.0)
    safety_gain_right = -safety_gain_left
    safety_target     = 0.75
    Win[3, bias_idx]  = safety_target
    Win[4, bias_idx]  = safety_target
    front_gain        = np.radians(-20.0)

    Wout[0, 0] = hit_turn
    Wout[0, 1] = heading_gain
    Wout[0, 2] = -heading_gain
    Wout[0, 3] = safety_gain_left
    Wout[0, 4] = safety_gain_right
    Wout[0, 5] = front_gain

    # ===== Internal state neurons (zero contribution to output) ==============

    # X6: Current clamped dtheta (computed in activation, no W needed).

    # X7: Direction accumulator.
    W[7, 7] = 1.0   # self-recurrence
    W[7, 6] = 1.0   # accumulate clamped dtheta

    # X8, X9: cos/sin -- computed in activation, no W/Win needed.

    # X10, X11: Position accumulators -- self-recurrence only.
    W[10, 10] = 1.0
    W[11, 11] = 1.0

    # X12: Hit signal.
    Win[12, hit_idx] = 1.0

    # X13: Latched heading correction (sample-and-hold).
    cal_gain = 1.0 / 0.173          # ~ 5.78 (empirically calibrated)
    W[13, 13]       =  1.0          # self-recurrence (latch)
    Win[13, L_idx]  = -cal_gain      # -(prox_L)
    Win[13, R_idx]  =  cal_gain      # +(prox_R)

    # X14: Instantaneous sensor contribution (same weights, NO recurrence).
    Win[14, L_idx]  = -cal_gain
    Win[14, R_idx]  =  cal_gain

    # ===== Reward-state circuit (from dummy3, remapped X6-X9 → X15-X18) =====

    K              = 0.005
    arm_from_energy = 1000.0
    arm_latch      = 10.0
    pulse_gain     = 100000.0
    pulse_thr      = 0.2
    arm_gate       = 1000.0
    latch_gain     = 10.0

    # X15: delayed copy of scaled energy
    Win[15, energy_idx] = K

    # X16: armed latch (gates X17 off during startup)
    W[16, 15] = arm_from_energy
    W[16, 16] = arm_latch

    # X17: reward detector, high whenever energy rises while armed
    Win[17, energy_idx] = pulse_gain * K
    W[17, 15]           = -pulse_gain
    W[17, 16]           = arm_gate
    Win[17, bias_idx]   = -(arm_gate + pulse_thr)

    # X18: latched is_rewarded flag
    W[18, 17] = latch_gain
    W[18, 18] = latch_gain

    # Build custom activation with Wout weights baked in
    wout_features = Wout[0, :6].copy()
    f = make_activation(speed, wout_features)

    model = Win, W, Wout, warmup, leak, f, g
    yield model


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from challenge_1 import train
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = 12345
    np.random.seed(seed)
    print("Training reflex player (single yield, should be instant)...")
    model = train(reflex_player, timeout=100)

    W_in, W, W_out, warmup, leak, f, g = model

    # Run validation on multiple seeds
    np.random.seed(seed)
    seeds = np.random.randint(0, 1_000_000, 4)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("reflex player validation (dummy4 + reward state)", fontsize=13)

    for run_idx, s in enumerate(seeds):
        np.random.seed(s)
        environment = Environment()
        bot = Bot()

        initial_dir = bot.direction
        initial_pos = np.array(bot.position, dtype=float)

        n_cam = bot.camera.resolution
        I = np.zeros((n_cam + 3, 1))
        X = np.zeros((1000, 1))

        bot.camera.update(bot.position, bot.direction,
                          environment.world, environment.colormap)

        true_dirs = [bot.direction]
        model_dirs = [np.pi / 2]  # placeholder for step 0
        true_pos = [np.array(bot.position, dtype=float)]
        model_pos = [np.array([0.5, 0.5])]
        reward_flags = [0.0]
        energy_history = [bot.energy]

        iteration = 0
        n_hits = 0
        while bot.energy > 0:
            I[:n_cam, 0] = 1 - bot.camera.depths
            I[n_cam:, 0] = bot.hit, bot.energy, 1.0
            X = (1 - leak) * X + leak * f(np.dot(W_in, I) + np.dot(W, X))
            O = np.dot(W_out, g(X))

            if iteration > warmup:
                bot.forward(O, environment, debug=False)
                n_hits += bot.hit

            # Direction from cos/sin neurons (zero lag)
            model_dir = float(np.arctan2(X[9, 0], X[8, 0]))
            model_px  = float(X[10, 0]) + 0.5
            model_py  = float(X[11, 0]) + 0.5

            true_dirs.append(bot.direction)
            model_dirs.append(model_dir)
            true_pos.append(np.array(bot.position, dtype=float))
            model_pos.append(np.array([model_px, model_py]))
            reward_flags.append(float(X[18, 0]))
            energy_history.append(bot.energy)

            iteration += 1

        true_dirs  = np.array(true_dirs)
        model_dirs = np.array(model_dirs)
        true_pos   = np.array(true_pos)
        model_pos  = np.array(model_pos)
        reward_flags = np.array(reward_flags)
        energy_history = np.array(energy_history)

        # Direction error (wrapped to [-pi, pi])
        dir_err = np.arctan2(np.sin(model_dirs - true_dirs),
                             np.cos(model_dirs - true_dirs))
        dir_err_abs = np.abs(dir_err)

        # Position error
        pos_err = np.sqrt(np.sum((model_pos - true_pos) ** 2, axis=1))

        # Latched correction
        correction_deg = np.degrees(float(X[13, 0]))

        print(f"\nRun {run_idx} (seed={s}):")
        print(f"  Steps: {iteration},  hits: {n_hits}")
        print(f"  Initial dir noise: {np.degrees(initial_dir - np.pi/2):.2f} deg")
        print(f"  Latched correction: {correction_deg:.2f} deg")
        print(f"  Direction error: mean={np.degrees(dir_err_abs.mean()):.3f} deg, "
              f"max={np.degrees(dir_err_abs.max()):.3f} deg, "
              f"final={np.degrees(dir_err_abs[-1]):.3f} deg")
        print(f"  Position error:  mean={pos_err.mean():.5f}, "
              f"max={pos_err.max():.5f}, "
              f"final={pos_err[-1]:.5f}")
        print(f"  Reward flag (X18): {float(X[18, 0]):.4f}")

        # --- Plot: direction ---
        ax = axes[run_idx, 0]
        ax.plot(np.degrees(true_dirs), label="true", alpha=0.7, linewidth=0.8)
        ax.plot(np.degrees(model_dirs), label="model", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("Direction (deg)")
        ax.set_title(f"Run {run_idx}  seed={s}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- Plot: position trajectory ---
        ax = axes[run_idx, 1]
        ax.plot(true_pos[:, 0], true_pos[:, 1],
                label="true", alpha=0.7, linewidth=0.8)
        ax.plot(model_pos[:, 0], model_pos[:, 1],
                label="model", alpha=0.7, linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_title("Position", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- Plot: errors ---
        ax = axes[run_idx, 2]
        ax.plot(np.degrees(dir_err_abs),
                label="dir err (deg)", alpha=0.7, linewidth=0.8)
        ax2 = ax.twinx()
        ax2.plot(pos_err, color="C1",
                 label="pos err", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("Dir error (deg)")
        ax2.set_ylabel("Pos error")
        ax.set_title("Errors", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- Plot: reward state & energy ---
        ax = axes[run_idx, 3]
        ax.plot(energy_history, label="energy", alpha=0.7, linewidth=0.8,
                color="C2")
        ax3 = ax.twinx()
        ax3.plot(reward_flags, label="X18 (rewarded)", alpha=0.7,
                 linewidth=0.8, color="C3")
        ax3.set_ylim(-0.1, 1.2)
        ax.set_ylabel("Energy")
        ax3.set_ylabel("Reward flag")
        ax.set_title("Reward state", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
        ax3.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reflex_validation.png", dpi=120)
    print("\nSaved reflex_validation.png")
