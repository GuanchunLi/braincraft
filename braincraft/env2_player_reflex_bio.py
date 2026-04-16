# Braincraft challenge - Reflex Bio Player for Environment 2
# Self-contained: no dependency on reflex player 2 or 3.

"""
Reflex Bio Player for Environment 2.

Based on reflex player 3 but with biologically plausible changes:

1. Color evidence gathering: Instead of copying raw colors and using
   np.isclose() in Python, dedicated neurons detect "close to blue":
   - Xi_hi neurons: relu_tanh(color_i - 3.5)  (fires for blue=4, red=5)
   - Xi_lo neurons: relu_tanh(color_i - 4.5)  (fires for red=5 only)
   - L_ev = relu_tanh(sum_left(Xi_hi) - 3*sum_left(Xi_lo))  (fires if
     any left ray sees blue)
   - R_ev analogous for right rays.
   Evidence accumulation and sign latch remain in the activation.

2. Front-gain modulation: Instead of computing front_sign * X5, two
   gated copies of the front-block neuron are used:
   - X5_pos = relu_tanh(Z5 + c*front_sign - c)  -> output weight +front_gain
   - X5_neg = relu_tanh(Z5 - c*front_sign - c)  -> output weight -front_gain
   Only one fires depending on the latched front_sign, with the bias
   correction (-c in the threshold) ensuring equivalence to the original.

Both changes introduce 1-step delays via the recurrent W matrix, which
is acceptable given evidence gathering takes multiple steps.
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_2 import Environment


# ── Activation helpers ──────────────────────────────────────────────
def identity(x):
    return x


def relu_tanh(x):
    """Thresholded activation used for sparse hand-crafted feature neurons."""
    x = np.tanh(x)
    return np.where(x > 0, x, 0)


# ── Shortcut circuit parameters (same as reflex2) ──────────────────
SHORTCUT_TURN    = -2.0
SIN_HORIZ_THR    = 0.35
SIN_VERT_THR     = 0.70
COUNTER_THR      = 60
NEAR_CENTER_THR  = 0.05
DRIFT_OFFSET     = 0.115
CORRIDOR_GAIN    = 5.0
CORRIDOR_SIN_THR = 0.30
CORRIDOR_STEPS   = 120
TURN_STEPS       = 18
APPROACH_STEPS   = 45
SC_TOTAL         = TURN_STEPS + APPROACH_STEPS

# ── Initial heading correction parameters ──────────────────────────
INIT_CORR_EPS    = 1e-3
INIT_CORR_CAP    = 0.15

# ── Bio-specific constants ─────────────────────────────────────────
COLOR_EVIDENCE_THRESHOLD = 2.0
FRONT_GAIN_MAG   = np.radians(20.0)
DEFAULT_SPEED    = 0.01
BLUE_HI_THR      = 3.5   # relu(color - 3.5): fires for blue(4), red(5)
BLUE_LO_THR      = 4.5   # relu(color - 4.5): fires for red(5) only
XI_LO_FACTOR     = 3.0   # inhibitory weight: 3 * Xi_lo cancels Xi_hi for red
GATE_C           = 1.0   # gating constant for X5_pos / X5_neg


# ── Neuron index layout ────────────────────────────────────────────
def _bio_indices(n_rays):
    """Return a dict of named neuron indices for the bio layout."""
    half = n_rays // 2
    xi_hi_start = 25
    xi_hi_stop  = xi_hi_start + n_rays       # 25..88
    xi_lo_start = xi_hi_stop
    xi_lo_stop  = xi_lo_start + n_rays        # 89..152
    l_ev        = xi_lo_stop                  # 153
    r_ev        = l_ev + 1                    # 154
    evidence    = r_ev + 1                    # 155
    front_sign  = evidence + 1                # 156
    x5_pos      = front_sign + 1              # 157
    x5_neg      = x5_pos + 1                  # 158
    bio_end     = x5_neg + 1                  # 159
    return dict(
        xi_hi_start=xi_hi_start, xi_hi_stop=xi_hi_stop,
        xi_lo_start=xi_lo_start, xi_lo_stop=xi_lo_stop,
        l_ev=l_ev, r_ev=r_ev,
        evidence=evidence, front_sign=front_sign,
        x5_pos=x5_pos, x5_neg=x5_neg,
        bio_end=bio_end, half=half,
    )


# ── Activation function ────────────────────────────────────────────
def make_activation(speed, wout_feature_weights, front_gain_mag, n_rays):
    """Build the custom per-neuron activation for the bio player."""

    a = np.radians(5)    # max turn per step
    wout = np.asarray(wout_feature_weights, dtype=float)
    idx = _bio_indices(n_rays)

    L_EV        = idx["l_ev"]
    R_EV        = idx["r_ev"]
    EVIDENCE    = idx["evidence"]
    FRONT_SIGN  = idx["front_sign"]
    X5_POS      = idx["x5_pos"]
    X5_NEG      = idx["x5_neg"]
    BIO_END     = idx["bio_end"]

    def f(x):
        # x has shape (n, 1) — challenge runners always pass column vectors.
        out = np.empty_like(x)

        # ── Standard feature neurons X0-X5 ──
        out[:6] = relu_tanh(x[:6])

        # ── Hit signal X12 ──
        out[12:13] = relu_tanh(x[12:13])

        # ── Reward circuit X15-X18 ──
        out[15:19] = relu_tanh(x[15:19])

        # ── Bio neurons: Xi_hi, Xi_lo, L_ev, R_ev, X5_pos, X5_neg ──
        out[25:BIO_END] = relu_tanh(x[25:BIO_END])

        # ── Read color evidence from network-computed L_ev / R_ev ──
        l_ev_val = float(out[L_EV, 0])
        r_ev_val = float(out[R_EV, 0])

        left_evidence  = 1.0 if l_ev_val > 0.1 and r_ev_val < 0.1 else 0.0
        right_evidence = 1.0 if r_ev_val > 0.1 and l_ev_val < 0.1 else 0.0

        # ── Evidence accumulator and front-sign latch ──
        evidence_prev   = float(x[EVIDENCE, 0])
        front_sign_prev = float(x[FRONT_SIGN, 0])

        if abs(front_sign_prev) < 0.5:
            evidence_now = evidence_prev + right_evidence - left_evidence
            if evidence_now >= COLOR_EVIDENCE_THRESHOLD:
                front_sign_val = 1.0
            elif evidence_now <= -COLOR_EVIDENCE_THRESHOLD:
                front_sign_val = -1.0
            else:
                front_sign_val = 0.0
        else:
            evidence_now   = evidence_prev
            front_sign_val = front_sign_prev

        out[EVIDENCE, 0]   = evidence_now
        out[FRONT_SIGN, 0] = front_sign_val

        # ── Gated front-block output from X5_pos / X5_neg ──
        x5_pos_val = float(out[X5_POS, 0])
        x5_neg_val = float(out[X5_NEG, 0])

        # ── Lagged heading state ──
        val7 = float(x[7, 0])
        x13  = float(x[13, 0])
        x14  = float(x[14, 0])

        if abs(val7) < 1e-8:
            correction = x13
            hold_val   = x13
        else:
            correction = x13 - x14
            hold_val   = correction

        theta_lagged = val7 + np.pi / 2 + correction
        sin_lagged   = np.sin(theta_lagged)

        # ── Initial heading correction X23/X24 ──
        x23_prev = float(x[23, 0])
        x24_prev = float(x[24, 0])

        if x24_prev < 0.5:
            init_remaining = float(
                np.clip(-correction, -INIT_CORR_CAP, INIT_CORR_CAP)
            )
            init_dtheta = 0.0
            init_correction_active = False
            x23_new = init_remaining
        else:
            init_remaining = x23_prev
            if abs(init_remaining) > INIT_CORR_EPS:
                init_dtheta = float(np.clip(init_remaining, -a, a))
                init_correction_active = True
                x23_new = init_remaining - init_dtheta
            else:
                init_dtheta = 0.0
                init_correction_active = False
                x23_new = 0.0

        out[23, 0] = x23_new
        out[24, 0] = 1.0

        # ── Step counter X19 ──
        x10_prev     = float(x[10, 0])
        x19_val      = float(x[19, 0])
        near_center  = abs(x10_prev) < NEAR_CENTER_THR
        heading_vert = abs(sin_lagged) > SIN_VERT_THR

        if near_center and heading_vert:
            counter = 0.0
        else:
            counter = x19_val + 1.0
        out[19, 0] = counter

        # ── Shortcut steering X20 ──
        is_rewarded   = float(out[18, 0]) > 0.5
        heading_horiz = abs(sin_lagged) < SIN_HORIZ_THR
        front_clear   = float(out[5, 0]) < 0.1
        enough_steps  = x19_val > COUNTER_THR
        cos_lagged    = np.cos(theta_lagged)
        if abs(cos_lagged) > 0.5:
            trig_offset = -DRIFT_OFFSET * np.sign(cos_lagged)
        else:
            trig_offset = 0.0
        near_corr = abs(x10_prev - trig_offset) < NEAR_CENTER_THR

        x22_prev = float(x[22, 0]) if x.shape[0] > 22 else 0.0
        trigger = (
            is_rewarded
            and heading_horiz
            and front_clear
            and enough_steps
            and near_corr
            and x22_prev < 0.5
        )

        if trigger:
            sc_countdown = float(SC_TOTAL)
        elif x22_prev > 0.5:
            sc_countdown = x22_prev - 1.0
        else:
            sc_countdown = 0.0

        is_turning     = sc_countdown > float(APPROACH_STEPS) + 0.5
        is_approaching = sc_countdown > 0.5 and not is_turning

        x11_prev = float(x[11, 0])
        if abs(x11_prev) > 0.01:
            turn_toward = -np.sign(cos_lagged) * np.sign(x11_prev)
        else:
            turn_toward = -1.0

        # ── Compute O_features ──
        O_main     = float(wout @ out[:6, 0])
        O_no_front = float(wout[:5] @ out[:5, 0])

        O_features = O_main + front_gain_mag * x5_pos_val - front_gain_mag * x5_neg_val

        out[22, 0] = sc_countdown

        if init_correction_active:
            x20_val = init_dtheta - O_features
        elif is_turning:
            x20_val = abs(SHORTCUT_TURN) * turn_toward
        elif is_approaching:
            x20_val = -O_no_front
        else:
            x20_val = 0.0
        out[20, 0] = x20_val

        O_now = O_features + x20_val
        dtheta_now = np.clip(O_now, -a, a)
        out[6, 0] = dtheta_now

        # ── Direction accumulator X7 (pass-through) ──
        out[7:8] = x[7:8]

        # ── Current heading ──
        theta_now = val7 + np.pi / 2 + correction + dtheta_now
        cos_now   = np.cos(theta_now)
        sin_now   = np.sin(theta_now)
        out[8, 0] = cos_now
        out[9, 0] = sin_now

        # ── Position accumulators X10-X11 (gated by hit) ──
        hit_val = float(x[12, 0])
        if hit_val < 0.5:
            out[10, 0] = float(x[10, 0]) + speed * cos_now
            out[11, 0] = float(x[11, 0]) + speed * sin_now
        else:
            out[10, 0] = float(x[10, 0])
            out[11, 0] = float(x[11, 0])

        # ── Correction neurons X13-X14 ──
        out[13, 0] = hold_val
        out[14, 0] = x14

        # ── Remaining neurons ──
        if x.shape[0] > BIO_END:
            out[BIO_END:] = relu_tanh(x[BIO_END:])

        return out

    return f


# ── Player builder ──────────────────────────────────────────────────
def reflex_bio_player():
    """Build the reflex bio player for env2 (self-contained)."""

    bot = Bot()
    n = 1000
    p = bot.camera.resolution          # 64
    warmup = 0
    leak = 1.0
    g = identity

    n_inputs = 2 * p + 3               # 131
    Win  = np.zeros((n, n_inputs))
    W    = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx    = 2 * p                  # 128
    energy_idx = 2 * p + 1              # 129
    bias_idx   = 2 * p + 2              # 130

    speed = bot.speed                   # 0.01
    n_rays = p                          # 64
    idx = _bio_indices(n_rays)

    # ── Feature neurons (same as reflex2) ───────────────────────────
    L_idx          = 20
    R_idx          = 63 - L_idx         # 43
    left_side_idx  = 11
    right_side_idx = 63 - left_side_idx # 52

    Win[0, hit_idx]        = 1.0        # X0: hit
    Win[1, L_idx]          = 1.0        # X1: prox[L]
    Win[2, R_idx]          = 1.0        # X2: prox[R]
    Win[3, left_side_idx]  = -1.0       # X3: safety L
    Win[4, right_side_idx] = -1.0       # X4: safety R

    C1_idx, C2_idx = 31, 32
    front_thr      = 1.4
    Win[5, C1_idx]   = 1.0
    Win[5, C2_idx]   = 1.0
    Win[5, bias_idx] = -front_thr       # X5: front-block (original)

    TANH1 = np.tanh(1.0)

    hit_turn          = np.radians(-10.0) / TANH1
    heading_gain      = np.radians(-40)
    safety_gain_left  = np.radians(-20.0)
    safety_gain_right = -safety_gain_left
    safety_target     = 0.75
    Win[3, bias_idx]  = safety_target
    Win[4, bias_idx]  = safety_target
    front_gain        = np.radians(-20.0)   # not used in Wout but kept for reference

    Wout[0, 0] = hit_turn
    Wout[0, 1] = heading_gain
    Wout[0, 2] = -heading_gain
    Wout[0, 3] = safety_gain_left
    Wout[0, 4] = safety_gain_right
    Wout[0, 5] = 0.0   # original X5 removed from output; gated copies used instead

    # ── Internal state neurons ──────────────────────────────────────
    W[7, 7] = 1.0      # X7: direction accumulator
    W[7, 6] = 1.0
    W[10, 10] = 1.0    # X10: position x
    W[11, 11] = 1.0    # X11: position y
    Win[12, hit_idx] = 1.0  # X12: hit signal

    # X13: latched heading correction
    cal_gain = 1.0 / 0.173
    W[13, 13]      =  1.0
    Win[13, L_idx] = -cal_gain
    Win[13, R_idx] =  cal_gain

    # X14: instantaneous sensor contribution
    Win[14, L_idx] = -cal_gain
    Win[14, R_idx] =  cal_gain

    # ── Reward circuit ──────────────────────────────────────────────
    K               = 0.005
    arm_from_energy = 1000.0
    arm_latch       = 10.0
    pulse_gain      = 100000.0
    pulse_thr       = 0.2
    arm_gate        = 1000.0
    latch_gain      = 10.0

    Win[15, energy_idx] = K
    W[16, 15] = arm_from_energy
    W[16, 16] = arm_latch
    Win[17, energy_idx] = pulse_gain * K
    W[17, 15]           = -pulse_gain
    W[17, 16]           = arm_gate
    Win[17, bias_idx]   = -(arm_gate + pulse_thr)
    W[18, 17] = latch_gain
    W[18, 18] = latch_gain

    # ── Shortcut and state neurons ──────────────────────────────────
    W[19, 19] = 1.0    # X19: step counter
    W[20, 20] = 1.0    # X20: shortcut steering override
    Wout[0, 20] = 1.0
    W[21, 21] = 1.0    # X21: corridor-active latch
    W[22, 22] = 1.0    # X22: shortcut countdown
    W[23, 23] = 1.0    # X23: initial heading correction remainder
    W[24, 24] = 1.0    # X24: seeded flag

    # ══════════════════════════════════════════════════════════════════
    # BIO CHANGE (i): Color evidence gathering neurons
    # ══════════════════════════════════════════════════════════════════
    # Xi_hi[r] = relu_tanh(color_r - BLUE_HI_THR)
    #   fires for blue (4 -> 0.5) and red (5 -> 1.5), after tanh ~0.46/0.91
    # Xi_lo[r] = relu_tanh(color_r - BLUE_LO_THR)
    #   fires for red (5 -> 0.5 -> tanh~0.46) only
    for r in range(n_rays):
        color_input_col = p + r   # color channel for ray r (indices p..2p-1)
        # Xi_hi neuron
        Win[idx["xi_hi_start"] + r, color_input_col] = 1.0
        Win[idx["xi_hi_start"] + r, bias_idx]        = -BLUE_HI_THR
        # Xi_lo neuron
        Win[idx["xi_lo_start"] + r, color_input_col] = 1.0
        Win[idx["xi_lo_start"] + r, bias_idx]        = -BLUE_LO_THR

    # L_ev aggregates left half Xi_hi (+1) and Xi_lo (-3) via W
    half = idx["half"]
    for r in range(half):
        W[idx["l_ev"], idx["xi_hi_start"] + r] =  1.0
        W[idx["l_ev"], idx["xi_lo_start"] + r] = -XI_LO_FACTOR

    # R_ev aggregates right half
    for r in range(half, n_rays):
        W[idx["r_ev"], idx["xi_hi_start"] + r] =  1.0
        W[idx["r_ev"], idx["xi_lo_start"] + r] = -XI_LO_FACTOR

    # Evidence accumulator (self-recurrence; updates done in activation)
    W[idx["evidence"],   idx["evidence"]]   = 1.0
    # Front-sign latch (self-recurrence; updates done in activation)
    W[idx["front_sign"], idx["front_sign"]] = 1.0

    # ══════════════════════════════════════════════════════════════════
    # BIO CHANGE (ii): Gated front-block copies X5_pos / X5_neg
    # ══════════════════════════════════════════════════════════════════
    # X5_pos = relu_tanh(Z5 + c*front_sign - c)
    #   where Z5 = prox_C1 + prox_C2 - front_thr
    #   implemented as: Win gives (prox_C1 + prox_C2 - (front_thr + c)),
    #   W gives +c * front_sign_prev.
    # When sign=+1: pre-act = Z5 (original). When sign=-1: pre-act = Z5 - 2c.
    X5_POS = idx["x5_pos"]
    X5_NEG = idx["x5_neg"]
    FS     = idx["front_sign"]

    Win[X5_POS, C1_idx]   = 1.0
    Win[X5_POS, C2_idx]   = 1.0
    Win[X5_POS, bias_idx] = -(front_thr + GATE_C)
    W[X5_POS, FS]         = GATE_C

    Win[X5_NEG, C1_idx]   = 1.0
    Win[X5_NEG, C2_idx]   = 1.0
    Win[X5_NEG, bias_idx] = -(front_thr + GATE_C)
    W[X5_NEG, FS]         = -GATE_C

    # Output weights for the gated copies
    Wout[0, X5_POS] =  FRONT_GAIN_MAG
    Wout[0, X5_NEG] = -FRONT_GAIN_MAG

    # ── Build activation ────────────────────────────────────────────
    wout_features = Wout[0, :6].copy()
    f = make_activation(speed, wout_features, FRONT_GAIN_MAG, n_rays)

    model = Win, W, Wout, warmup, leak, f, g
    yield model


if __name__ == "__main__":
    import time
    from challenge_2 import evaluate, train

    seed = 12345
    np.random.seed(seed)
    print("Training reflex_bio player for env2 (single yield, instant)...")
    model = train(reflex_bio_player, timeout=100)

    W_in, W, W_out, warmup, leak, f, g = model

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score (distance): {score:.2f} +/- {std:.2f}")
