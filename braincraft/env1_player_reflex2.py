# Braincraft challenge — Reflex Player v2 (half-ring shortcut)
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3

"""
Reflex Player v2 — half-ring shortcut after first reward.

Extends the reflex player v1 with a shortcut steering circuit that
redirects the bot through the inner corridor after receiving the first
reward, creating a half-ring lap pattern instead of a full ring.

Behavior:
  1. First lap: full clockwise outer ring (identical to v1).
  2. After first reward encounter: at the next opportunity where
     the bot is heading horizontally near the corridor entrance,
     it turns right into the corridor, skipping half the ring.
  3. Subsequent laps: half-ring (one side + corridor), always
     visiting the same reward source.

New internal state neurons (additions to v1):

- X19: step counter since last corridor pass
       Increments each step; resets to 0 when heading is roughly
       vertical AND position is near the corridor center (|X10|<thr).
       Prevents immediate re-entry after exiting the corridor.

- X20: shortcut steering signal (latched)
       Activates (1.0) when ALL conditions are satisfied:
         (a) is_rewarded        — X18 > 0.5
         (b) heading horizontal — |sin(theta_lagged)| < 0.35
         (c) no front wall      — X5 < 0.1
         (d) enough steps       — X19 > 60
         (e) near corridor      — |X10| < 0.15
       Once triggered, maintained until heading becomes nearly vertical
       (|sin| > 0.85), ensuring a decisive ~60 deg right turn.
       Wout[0, 20] = -2.0  (strong right-turn override).
"""

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


# --------------- Shortcut circuit parameters --------------------------------
SHORTCUT_TURN    = -2.0    # Wout weight: strong right turn
SIN_HORIZ_THR   = 0.35    # |sin| < this  => heading "horizontal"
SIN_VERT_THR    = 0.70    # |sin| > this  => heading "vertical" (counter reset)
SIN_MAINTAIN_THR = 0.50   # latch release (~30 deg turn, less corridor drift)
COUNTER_THR     = 60      # min steps before shortcut can trigger
NEAR_CENTER_THR = 0.12    # |X10 - offset| < this  => near corridor
DRIFT_OFFSET    = 0.05    # pre-shift trigger opposite to travel direction
# ----------------------------------------------------------------------------


def make_activation(speed, wout_feature_weights, wout_shortcut_weight=0.0):
    """Build a custom per-neuron activation with direction/position tracking,
    reward detection, and shortcut steering.

    Parameters
    ----------
    speed : float
        Bot speed (0.01).
    wout_feature_weights : array, shape (6,)
        Wout[0, 0:6] — output weights for the 6 feature neurons.
    wout_shortcut_weight : float
        Wout[0, 20] — output weight for the shortcut steering neuron.
    """
    a = np.radians(5)  # +/-5 degrees clamp limit
    wout = np.asarray(wout_feature_weights, dtype=float)
    wout_sc = float(wout_shortcut_weight)

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

        # Helpers for indexing into 1D or (n,1) arrays
        def _g(arr, i):
            return float(arr[i, 0]) if is2d else float(arr[i])

        def _s(arr, i, v):
            if is2d:
                arr[i, 0] = v
            else:
                arr[i] = v

        out = np.empty_like(x)

        # === Feature neurons X0-X5 (relu_tanh, drive output) ===
        out[:6] = relu_tanh(x[:6])

        # === Hit signal X12 (relu_tanh, for position gating) ===
        out[12:13] = relu_tanh(x[12:13])

        # === Reward circuit X15-X18 (relu_tanh) ===
        out[15:19] = relu_tanh(x[15:19])

        # === Lagged direction (from previous steps, before current turn) ===
        val7 = _g(x, 7)   # cumulative dtheta (1-step lag)
        x13  = _g(x, 13)
        x14  = _g(x, 14)

        if abs(val7) < 1e-8:
            # Step 0: sample correction from sensor asymmetry
            correction = x13
            hold_val   = x13
        else:
            # Step 1+: recover latched correction
            correction = x13 - x14
            hold_val   = correction

        theta_lagged = val7 + np.pi / 2 + correction
        sin_lagged = np.sin(theta_lagged)

        # === Step counter X19 ===
        x10_prev = _g(x, 10)          # previous x-displacement
        x19_val  = _g(x, 19)          # previous counter value
        near_center  = abs(x10_prev) < NEAR_CENTER_THR
        heading_vert = abs(sin_lagged) > SIN_VERT_THR

        if near_center and heading_vert:
            counter = 0.0              # in the corridor — reset
        else:
            counter = x19_val + 1.0    # increment
        _s(out, 19, counter)

        # === Shortcut steering X20 (latched) ===
        is_rewarded   = _g(out, 18) > 0.5
        heading_horiz = abs(sin_lagged) < SIN_HORIZ_THR
        front_clear   = _g(out, 5) < 0.1
        enough_steps  = x19_val > COUNTER_THR
        # Direction-aware trigger: offset opposite to travel to compensate
        # for lateral drift during the turn
        cos_lagged = np.cos(theta_lagged)
        trig_offset = -DRIFT_OFFSET * np.sign(cos_lagged) if abs(cos_lagged) > 0.5 else 0.0
        near_corr   = abs(x10_prev - trig_offset) < NEAR_CENTER_THR

        x20_prev = _g(x, 20)          # previous latch state
        trigger  = (is_rewarded and heading_horiz and front_clear
                    and enough_steps and near_corr)
        maintain = x20_prev > 0.5 and abs(sin_lagged) < SIN_MAINTAIN_THR

        sc_val = 1.0 if (trigger or maintain) else 0.0
        _s(out, 20, sc_val)

        # === Compute dtheta including shortcut contribution ===
        if is2d:
            O_now = float(wout @ out[:6, 0]) + wout_sc * sc_val
        else:
            O_now = float(wout @ out[:6]) + wout_sc * sc_val
        dtheta_now = np.clip(O_now, -a, a)
        _s(out, 6, dtheta_now)

        # === Direction accumulator X7 (identity) ===
        out[7:8] = x[7:8]             # pass-through lagged sum

        # === Zero-lag direction with correction ===
        theta_now = val7 + np.pi / 2 + correction + dtheta_now
        cos_now = np.cos(theta_now)
        sin_now = np.sin(theta_now)
        _s(out, 8, cos_now)
        _s(out, 9, sin_now)

        # === Position accumulators X10-X11 (gated by hit) ===
        hit_val = _g(x, 12)
        if hit_val < 0.5:
            _s(out, 10, _g(x, 10) + speed * cos_now)
            _s(out, 11, _g(x, 11) + speed * sin_now)
        else:
            _s(out, 10, _g(x, 10))
            _s(out, 11, _g(x, 11))

        # === Correction neurons X13-X14 ===
        _s(out, 13, hold_val)
        _s(out, 14, x14)              # identity pass-through

        # === Remaining neurons X21-X999 (relu_tanh, all zero) ===
        if x.shape[0] > 21:
            out[21:] = relu_tanh(x[21:])

        return out

    return f


def reflex2_player():
    """Build reflex player v2 — clockwise wall-following with corridor
    shortcut after first reward."""

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

    # ===== Feature neurons (identical to reflex v1) ==========================
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

    # ===== Gains (identical to reflex v1) ====================================
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

    # ===== Internal state neurons (same as reflex v1) ========================
    # X6:  current clamped dtheta (computed in activation)
    # X7:  direction accumulator
    W[7, 7] = 1.0
    W[7, 6] = 1.0

    # X8, X9: cos/sin (computed in activation)
    # X10, X11: position accumulators
    W[10, 10] = 1.0
    W[11, 11] = 1.0

    # X12: hit signal
    Win[12, hit_idx] = 1.0

    # X13: latched heading correction (sample-and-hold)
    cal_gain = 1.0 / 0.173
    W[13, 13]       =  1.0
    Win[13, L_idx]  = -cal_gain
    Win[13, R_idx]  =  cal_gain

    # X14: instantaneous sensor contribution (no recurrence)
    Win[14, L_idx]  = -cal_gain
    Win[14, R_idx]  =  cal_gain

    # ===== Reward circuit (same as reflex v1) ================================
    K              = 0.005
    arm_from_energy = 1000.0
    arm_latch      = 10.0
    pulse_gain     = 100000.0
    pulse_thr      = 0.2
    arm_gate       = 1000.0
    latch_gain     = 10.0

    Win[15, energy_idx] = K
    W[16, 15] = arm_from_energy
    W[16, 16] = arm_latch
    Win[17, energy_idx] = pulse_gain * K
    W[17, 15]           = -pulse_gain
    W[17, 16]           = arm_gate
    Win[17, bias_idx]   = -(arm_gate + pulse_thr)
    W[18, 17] = latch_gain
    W[18, 18] = latch_gain

    # ===== NEW: Shortcut circuit =============================================
    # X19: step counter (self-recurrence; logic in activation)
    W[19, 19] = 1.0

    # X20: shortcut steering latch (self-recurrence; logic in activation)
    W[20, 20] = 1.0
    Wout[0, 20] = SHORTCUT_TURN

    # Build custom activation with all relevant weights baked in
    wout_features = Wout[0, :6].copy()
    f = make_activation(speed, wout_features,
                        wout_shortcut_weight=SHORTCUT_TURN)

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
    print("Training reflex2 player (single yield, instant)...")
    model = train(reflex2_player, timeout=100)

    W_in, W, W_out, warmup, leak, f, g = model

    # Run validation on multiple seeds
    np.random.seed(seed)
    seeds = np.random.randint(0, 1_000_000, 4)

    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    fig.suptitle("Reflex Player v2 — half-ring shortcut validation", fontsize=13)

    for run_idx, s in enumerate(seeds):
        np.random.seed(s)
        environment = Environment()
        bot = Bot()

        n_cam = bot.camera.resolution
        I = np.zeros((n_cam + 3, 1))
        X = np.zeros((1000, 1))

        bot.camera.update(bot.position, bot.direction,
                          environment.world, environment.colormap)

        true_pos = [np.array(bot.position, dtype=float)]
        energy_history = [bot.energy]
        reward_flags = [0.0]
        shortcut_flags = [0.0]
        counter_vals = [0.0]

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

            true_pos.append(np.array(bot.position, dtype=float))
            energy_history.append(bot.energy)
            reward_flags.append(float(X[18, 0]))
            shortcut_flags.append(float(X[20, 0]))
            counter_vals.append(float(X[19, 0]))
            iteration += 1

        true_pos = np.array(true_pos)
        energy_history = np.array(energy_history)
        reward_flags = np.array(reward_flags)
        shortcut_flags = np.array(shortcut_flags)
        counter_vals = np.array(counter_vals)

        print(f"\nRun {run_idx} (seed={s}): {iteration} steps, "
              f"{n_hits} hits, reward={float(X[18,0]):.2f}")

        # --- Plot: trajectory on world map ---
        ax = axes[run_idx, 0]
        extent = [0, 1, 0, 1]
        ax.imshow(environment.world_rgb, origin="lower", extent=extent,
                  interpolation="nearest")
        t = np.linspace(0, 1, len(true_pos))
        ax.scatter(true_pos[:, 0], true_pos[:, 1], c=t, cmap="viridis",
                   s=1, zorder=10)
        ax.plot(true_pos[0, 0], true_pos[0, 1], "o", color="lime",
                markersize=8, markeredgecolor="black", zorder=20)
        ax.plot(true_pos[-1, 0], true_pos[-1, 1], "X", color="red",
                markersize=8, markeredgecolor="black", zorder=20)
        # Mark shortcut activation points
        sc_on = np.where(np.diff(shortcut_flags) > 0.5)[0] + 1
        if len(sc_on):
            ax.scatter(true_pos[sc_on, 0], true_pos[sc_on, 1],
                       marker="v", color="magenta", s=60, zorder=25,
                       label="shortcut")
            ax.legend(fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"Run {run_idx}  seed={s}", fontsize=9)

        # --- Plot: energy ---
        ax = axes[run_idx, 1]
        ax.plot(energy_history, linewidth=0.8, color="C2")
        ax.set_ylabel("Energy")
        ax.set_title("Energy", fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Plot: reward flag ---
        ax = axes[run_idx, 2]
        ax.plot(reward_flags, linewidth=0.8, color="C3")
        ax.set_ylim(-0.1, 1.2)
        ax.set_ylabel("X18")
        ax.set_title("Reward flag", fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Plot: shortcut flag ---
        ax = axes[run_idx, 3]
        ax.plot(shortcut_flags, linewidth=0.8, color="C4")
        ax.set_ylim(-0.1, 1.2)
        ax.set_ylabel("X20")
        ax.set_title("Shortcut active", fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Plot: step counter ---
        ax = axes[run_idx, 4]
        ax.plot(counter_vals, linewidth=0.8, color="C5")
        ax.axhline(COUNTER_THR, color="red", linestyle="--",
                    linewidth=0.8, alpha=0.5)
        ax.set_ylabel("X19")
        ax.set_title("Step counter", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reflex2_validation.png", dpi=120)
    print("\nSaved reflex2_validation.png")
