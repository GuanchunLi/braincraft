# Braincraft challenge - Reflex Bio Player 2 for Environment 2
# Self-contained: no dependency on other reflex players.

"""
Reflex Bio Player 2 for Environment 2.

Constants use snake_case; hidden-state aliases use UPPER_SNAKE.

This is the pointwise-activation version of the env2 bio controller.
After construction, all runtime logic is carried by fixed scalar
activations plus weights in Win and W:

    X(t+1) = f(Win @ I(t) + W @ X(t))
    O(t+1) = Wout @ g(X(t+1))

with identity readout g(x) = x.

The dense bio2 layout is grouped as:

  0..4   reflex features
  5..8   dtheta, dir_accum, pos_x, pos_y
  9..12  initial-heading correction latch
  13..19 reward and shortcut state
  20..39 trig, corridor, and shortcut-trigger helpers
  40..57 shortcut-phase, front-block, and color-evidence helpers
  58..   xi_blue ray bank

The trig pair preserves the established pos_x / pos_y frame while using
only the sin activation, and the shortcut state machine is expressed
with thresholded helper neurons instead of Python control flow.
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_2 import Environment


# ── Shortcut circuit parameters ────────────────────────────────────
shortcut_turn = -2.0
sin_horiz_thr = 0.35
near_c_thr = 0.05
drift_offset = 0.175
turn_steps = 18
approach_steps = 45
sc_total = turn_steps + approach_steps

# ── Bio-specific constants ─────────────────────────────────────────
color_evidence_thr = 2.0
front_gain_mag   = np.radians(20.0)
gate_c           = 1.0

# Sharp-threshold gain for crisp AND / OR / latch circuits.
k_sharp = 50.0     # large enough to saturate tanh near |z|>0.1
step_a  = np.radians(5.0)


# ── Neuron index layout ────────────────────────────────────────────
def _bio2_indices(n_rays):
    """Sequential neuron indices for the bio2 layout."""
    idx = {
        "hit_feat": 0,
        "prox_left": 1,
        "prox_right": 2,
        "safe_left": 3,
        "safe_right": 4,
        "dtheta": 5,
        "dir_accum": 6,
        "pos_x": 7,
        "pos_y": 8,
        "head_corr": 9,
        "seeded_flag": 10,
        "seed_pos": 11,
        "seed_neg": 12,
        "energy_ramp": 13,
        "armed_latch": 14,
        "reward_pulse": 15,
        "reward_latch": 16,
        "sc_countdown": 17,
        "shortcut_steer": 18,
        "init_impulse": 19,
        "cos_n": 20,
        "sin_n": 21,
        "sin_sq": 22,
        "cos_pos": 23,
        "cos_neg": 24,
        "y_pos": 25,
        "y_neg": 26,
        "cos_big_pos": 27,
        "cos_big_neg": 28,
        "cos_small": 29,
        "near_c": 30,
        "near_e": 31,
        "near_w": 32,
        "ncr_e": 33,
        "ncr_w": 34,
        "ncr_c": 35,
        "near_cr": 36,
        "heading_horiz": 37,
        "front_clear": 38,
        "trig_sc": 39,
        "on_countdown": 40,
        "is_turn": 41,
        "is_app": 42,
        "cy_pp": 43,
        "cy_pn": 44,
        "cy_np": 45,
        "cy_nn": 46,
        "front_block_pos": 47,
        "front_block_neg": 48,
        "l_ev": 49,
        "r_ev": 50,
        "dleft": 51,
        "dright": 52,
        "evidence": 53,
        "trig_pos": 54,
        "trig_neg": 55,
        "fs_pos": 56,
        "fs_neg": 57,
    }
    idx["xi_blue_start"] = 58
    idx["xi_blue_stop"] = idx["xi_blue_start"] + n_rays
    idx["half"] = n_rays // 2
    idx["bio_end"] = idx["xi_blue_stop"]
    return idx


# ── Pointwise activation function ─────────────────────────────────
def make_activation(a, idx):
    """Return the per-neuron pointwise activation.

    Each out[i] depends only on x[i].  The activation type for neuron i
    is looked up from index arrays built once here.
    """
    id_list = [
        idx["dir_accum"],
        idx["pos_x"],
        idx["pos_y"],
        idx["head_corr"],
        idx["shortcut_steer"],
        idx["init_impulse"],
        idx["evidence"],
        idx["l_ev"],
        idx["r_ev"],
    ]
    sin_list = [idx["cos_n"], idx["sin_n"]]
    square_list = [idx["sin_sq"]]
    relu_list = [idx["energy_ramp"], idx["sc_countdown"]]
    bump_list = [idx["near_c"], idx["near_e"], idx["near_w"]]
    bump_list.extend(range(idx["xi_blue_start"], idx["xi_blue_stop"]))

    id_arr   = np.array(sorted(set(id_list)), dtype=int)
    sin_arr  = np.array(sorted(set(sin_list)), dtype=int)
    square_arr = np.array(sorted(set(square_list)), dtype=int)
    relu_arr = np.array(sorted(set(relu_list)), dtype=int)
    bump_arr = np.array(sorted(set(bump_list)), dtype=int)

    def f(x):
        # Default: relu_tanh.
        out = np.maximum(0.0, np.tanh(x))
        if id_arr.size:
            out[id_arr, 0] = x[id_arr, 0]
        if sin_arr.size:
            out[sin_arr, 0] = np.sin(x[sin_arr, 0])
        if square_arr.size:
            out[square_arr, 0] = x[square_arr, 0] ** 2
        if relu_arr.size:
            out[relu_arr, 0] = np.maximum(0.0, x[relu_arr, 0])
        if bump_arr.size:
            out[bump_arr, 0] = np.maximum(0.0, 1.0 - 4.0 * x[bump_arr, 0] ** 2)
        # dtheta : clip(z, -a, a)
        out[idx["dtheta"], 0] = float(np.clip(x[idx["dtheta"], 0], -a, a))
        return out

    return f


# ── Player builder ──────────────────────────────────────────────────
def reflex_bio2_player():
    """Build the bio2 reflex player for env2."""

    bot = Bot()
    n = 1000
    p = bot.camera.resolution          # 64
    warmup = 0
    leak = 1.0
    g = lambda x: x                    # identity readout

    n_inputs = 2 * p + 3
    Win  = np.zeros((n, n_inputs))
    W    = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx    = 2 * p                  # 128
    energy_idx = 2 * p + 1              # 129
    bias_idx   = 2 * p + 2              # 130

    speed = bot.speed                   # 0.01
    n_rays = p                          # 64
    idx = _bio2_indices(n_rays)
    a = step_a

    # Aliases.
    HIT_FEAT = idx["hit_feat"]
    PROX_LEFT = idx["prox_left"]
    PROX_RIGHT = idx["prox_right"]
    SAFE_LEFT = idx["safe_left"]
    SAFE_RIGHT = idx["safe_right"]
    DTHETA = idx["dtheta"]
    DIR_ACCUM = idx["dir_accum"]
    POS_X = idx["pos_x"]
    POS_Y = idx["pos_y"]
    HEAD_CORR = idx["head_corr"]
    SEEDED_FLAG = idx["seeded_flag"]
    ENERGY_RAMP = idx["energy_ramp"]
    ARMED_LATCH = idx["armed_latch"]
    REWARD_PULSE = idx["reward_pulse"]
    REWARD_LATCH = idx["reward_latch"]
    SC_COUNTDOWN = idx["sc_countdown"]
    L_EV     = idx["l_ev"]
    R_EV     = idx["r_ev"]
    DLEFT    = idx["dleft"]
    DRIGHT   = idx["dright"]
    EV       = idx["evidence"]
    TP       = idx["trig_pos"]
    TN       = idx["trig_neg"]
    FS_P     = idx["fs_pos"]
    FS_N     = idx["fs_neg"]
    FBP      = idx["front_block_pos"]
    FBN      = idx["front_block_neg"]
    SHORTCUT_STEER = idx["shortcut_steer"]
    INIT_IMPULSE   = idx["init_impulse"]
    COS_N    = idx["cos_n"]
    SIN_N    = idx["sin_n"]
    SIN_SQ   = idx["sin_sq"]
    COS_POS  = idx["cos_pos"]
    COS_NEG  = idx["cos_neg"]
    Y_POS    = idx["y_pos"]
    Y_NEG    = idx["y_neg"]
    NEAR_C   = idx["near_c"]
    HH       = idx["heading_horiz"]
    FC       = idx["front_clear"]
    TSC      = idx["trig_sc"]
    ONC      = idx["on_countdown"]
    IST      = idx["is_turn"]
    ISA      = idx["is_app"]
    SEEDP    = idx["seed_pos"]
    SEEDN    = idx["seed_neg"]
    COSBP    = idx["cos_big_pos"]
    COSBN    = idx["cos_big_neg"]
    NEAR_E   = idx["near_e"]
    NEAR_W   = idx["near_w"]
    NCR_E    = idx["ncr_e"]
    NCR_W    = idx["ncr_w"]
    COS_SMALL = idx["cos_small"]
    NCR_C    = idx["ncr_c"]
    NEAR_CR  = idx["near_cr"]
    CY_PP    = idx["cy_pp"]
    CY_PN    = idx["cy_pn"]
    CY_NP    = idx["cy_np"]
    CY_NN    = idx["cy_nn"]

    # ── Reflex feature neurons ───────────────────────────────────
    L_idx          = 20
    R_idx          = 63 - L_idx         # 43
    left_side_idx  = 11
    right_side_idx = 63 - left_side_idx # 52

    Win[HIT_FEAT, hit_idx] = 1.0
    Win[PROX_LEFT, L_idx] = 1.0
    Win[PROX_RIGHT, R_idx] = 1.0
    Win[SAFE_LEFT, left_side_idx] = -1.0
    Win[SAFE_RIGHT, right_side_idx] = -1.0

    # Approach-phase cancellation: silence reflex features 0..4 while
    # IS_APP is active, so O reduces to the front block plus shortcut steering.
    for reflex_idx in (HIT_FEAT, PROX_LEFT, PROX_RIGHT, SAFE_LEFT, SAFE_RIGHT):
        W[reflex_idx, ISA] = -k_sharp * 100.0

    C1_idx, C2_idx = 31, 32
    front_thr      = 1.4

    TANH1 = np.tanh(1.0)

    hit_turn          = np.radians(-10.0) / TANH1
    heading_gain      = np.radians(-40.0)
    safety_gain_left  = np.radians(-20.0)
    safety_gain_right = -safety_gain_left
    safety_target     = 0.75
    Win[SAFE_LEFT, bias_idx] = safety_target
    Win[SAFE_RIGHT, bias_idx] = safety_target

    Wout[0, HIT_FEAT] = hit_turn
    Wout[0, PROX_LEFT] = heading_gain
    Wout[0, PROX_RIGHT] = -heading_gain
    Wout[0, SAFE_LEFT] = safety_gain_left
    Wout[0, SAFE_RIGHT] = safety_gain_right
    Wout[0, SHORTCUT_STEER] = 1.0
    Wout[0, INIT_IMPULSE] = 1.0
    # dtheta stores the clipped one-step-lagged steering command.
    # We fill W[dtheta, :] after Wout is finalized so
    # z_dtheta(t+1) = Wout @ X(t) = O(t).
    # Direction accumulator.
    W[DIR_ACCUM, DIR_ACCUM] = 1.0
    W[DIR_ACCUM, DTHETA] = 1.0
    # Position accumulators.
    # No hit-gating: drift during transient hits is negligible.
    # Under trig option 2A both neurons use the sin activation, but the
    # biases preserve the current downstream frame exactly.
    W[POS_X, POS_X] = 1.0
    W[POS_X, COS_N] = speed
    W[POS_Y, POS_Y] = 1.0
    W[POS_Y, SIN_N] = speed
    # Initial heading correction latch.
    # Bio1 latches the initial correction from step 0 onward. Here
    # head_corr self-recurs and is seeded by seed_pos/seed_neg before
    # seeded_flag turns on. That introduces a one-step lag vs bio1:
    # head_corr(0) = 0 and head_corr(t) = current_corr(0) for t >= 1.
    cal_gain = 1.0 / 0.173
    W[HEAD_CORR, HEAD_CORR] = 1.0
    W[HEAD_CORR, SEEDP] = 1.0
    W[HEAD_CORR, SEEDN] = -1.0
    # Reward circuit.
    arm_from_energy = 5.0
    arm_latch       = 10.0
    pulse_gain      = 500.0
    pulse_thr       = 0.2
    arm_gate        = 1000.0
    latch_gain      = 10.0

    Win[ENERGY_RAMP, energy_idx] = 1.0
    W[ARMED_LATCH, ENERGY_RAMP] = arm_from_energy
    W[ARMED_LATCH, ARMED_LATCH] = arm_latch
    Win[REWARD_PULSE, energy_idx] = pulse_gain
    W[REWARD_PULSE, ENERGY_RAMP] = -pulse_gain
    W[REWARD_PULSE, ARMED_LATCH] = arm_gate
    Win[REWARD_PULSE, bias_idx] = -(arm_gate + pulse_thr)
    W[REWARD_LATCH, REWARD_PULSE] = latch_gain
    W[REWARD_LATCH, REWARD_LATCH] = latch_gain
    # seeded_flag saturates to 1 at step 1.
    Win[SEEDED_FLAG, bias_idx] = 10.0
    # Shortcut outputs.
    # Route shortcut steering and the one-shot init impulse through
    # separate fixed slots so debug/docs can name them directly.
    W[SHORTCUT_STEER, CY_PN] =  abs(shortcut_turn)   # cos+, y-  → +2
    W[SHORTCUT_STEER, CY_NP] =  abs(shortcut_turn)   # cos-, y+  → +2
    W[SHORTCUT_STEER, CY_PP] = -abs(shortcut_turn)   # cos+, y+  → -2
    W[SHORTCUT_STEER, CY_NN] = -abs(shortcut_turn)   # cos-, y-  → -2

    # One-shot init correction: SEED_{POS,NEG} fire only before seeded_flag
    # latches, so INIT_IMPULSE contributes -current_corr for one step.
    W[INIT_IMPULSE, SEEDP] = -1.0
    W[INIT_IMPULSE, SEEDN] =  1.0
    # sc_countdown(t+1) = sc_countdown(t) - 1 + (sc_total + 1) * trig_sc(t).
    W[SC_COUNTDOWN, SC_COUNTDOWN] = 1.0
    Win[SC_COUNTDOWN, bias_idx] = -1.0
    W[SC_COUNTDOWN, TSC] = float(sc_total) + 1.0

    # SEED_POS / SEED_NEG: capture +/- part of current_corr ONLY when
    # seeded_flag(t) is 0 (unseeded). After seeded_flag latches to 1 at step 1+,
    # a huge negative gate suppresses them.
    # SEED_POS fires when current_corr > 0 AND not yet seeded.
    Win[SEEDP, L_idx] = -cal_gain
    Win[SEEDP, R_idx] =  cal_gain
    W[SEEDP, SEEDED_FLAG] = -1.0e3        # when seeded_flag=1, z ≈ -1000 → rt = 0

    Win[SEEDN, L_idx] =  cal_gain
    Win[SEEDN, R_idx] = -cal_gain
    W[SEEDN, SEEDED_FLAG] = -1.0e3
    # xi_blue color evidence rays.
    for r in range(n_rays):
        color_input_col = p + r
        Win[idx["xi_blue_start"] + r, color_input_col] = 1.0
        Win[idx["xi_blue_start"] + r, bias_idx] = -4.0

    half = idx["half"]
    for r in range(half):
        W[L_EV, idx["xi_blue_start"] + r] = 1.0
    for r in range(half, n_rays):
        W[R_EV, idx["xi_blue_start"] + r] = 1.0
    # Gated delta pulses.
    # Fire only when one side dominates AND front sign NOT yet latched.
    # With k_sharp=50 gain and rt, saturates to ~1 when L/R difference
    # exceeds a margin.  The latch subtraction zeros them out once
    # FS_POS or FS_NEG is 1.
    W[DLEFT, L_EV]  =  k_sharp
    W[DLEFT, R_EV]  = -k_sharp
    W[DLEFT, FS_P]  = -k_sharp * 10
    W[DLEFT, FS_N]  = -k_sharp * 10
    Win[DLEFT, bias_idx] = -0.2 * k_sharp

    W[DRIGHT, L_EV] = -k_sharp
    W[DRIGHT, R_EV] =  k_sharp
    W[DRIGHT, FS_P] = -k_sharp * 10
    W[DRIGHT, FS_N] = -k_sharp * 10
    Win[DRIGHT, bias_idx] = -0.2 * k_sharp
    # evidence: signed integrator.
    W[EV, EV]     = 1.0
    W[EV, DRIGHT] = 1.0
    W[EV, DLEFT]  = -1.0
    # Trigger thresholds.
    # bio1 uses `if evidence_now >= threshold`; rt(K*(EV-thr)) for EV==thr
    # gives rt(0)=0, so we lower the threshold by 0.5 so TP fires at
    # EV==threshold (DLEFT/DRIGHT saturate to ~±1, so EV is integer-ish).
    W[TP, EV] =  k_sharp
    Win[TP, bias_idx] = -k_sharp * (color_evidence_thr - 0.5)
    W[TN, EV] = -k_sharp
    Win[TN, bias_idx] = -k_sharp * (color_evidence_thr - 0.5)
    # Front-sign latches.
    W[FS_P, FS_P] = k_sharp
    W[FS_P, TP]   = k_sharp
    W[FS_N, FS_N] = k_sharp
    W[FS_N, TN]   = k_sharp
    # front_block_pos / front_block_neg.
    Win[FBP, C1_idx] = 1.0
    Win[FBP, C2_idx] = 1.0
    Win[FBP, bias_idx] = -(front_thr + gate_c)
    W[FBP, FS_P] = gate_c
    W[FBP, FS_N] = -gate_c

    Win[FBN, C1_idx] = 1.0
    Win[FBN, C2_idx] = 1.0
    Win[FBN, bias_idx] = -(front_thr + gate_c)
    W[FBN, FS_P] = -gate_c
    W[FBN, FS_N] = gate_c

    Wout[0, FBP] = front_gain_mag
    Wout[0, FBN] = -front_gain_mag
    # Trig neurons (sin-only activation, preserved output frame).
    for idx_trig in (COS_N, SIN_N):
        W[idx_trig, DIR_ACCUM] = 1.0
        W[idx_trig, HEAD_CORR] = 1.0
        W[idx_trig, DTHETA] = 1.0
    Win[COS_N, bias_idx] = np.pi
    Win[SIN_N, bias_idx] = np.pi / 2
    # Magnitude/sign helpers.
    W[SIN_SQ, SIN_N] = 1.0

    # COS_POS/COS_NEG only feed quadrant sign logic, so saturation is fine.
    W[COS_POS, COS_N] =  k_sharp
    W[COS_NEG, COS_N] = -k_sharp

    # Y_POS/Y_NEG feed quadrant sign logic as sign(y) - saturation OK.
    W[Y_POS, POS_Y] = k_sharp
    W[Y_NEG, POS_Y] = -k_sharp

    # Bump support is |z| < 0.5, so use z = value / (2*thr).
    bump_scale = 1.0 / (2.0 * near_c_thr)
    W[NEAR_C, POS_X] = bump_scale

    # HH reads the exact squared sine magnitude.
    W[HH, SIN_SQ] = -k_sharp / (sin_horiz_thr ** 2)
    Win[HH, bias_idx] = k_sharp

    # FRONT_CLEAR: front_block_pos + front_block_neg < 0.1.
    W[FC, FBP] = -k_sharp
    W[FC, FBN] = -k_sharp
    Win[FC, bias_idx] =  k_sharp * 0.1
    # near_cr helpers.
    # COS_BIG_POS: cos > 0.5 (sharp).
    W[COSBP, COS_N] =  k_sharp
    Win[COSBP, bias_idx] = -0.5 * k_sharp
    # COS_BIG_NEG: cos < -0.5.
    W[COSBN, COS_N] = -k_sharp
    Win[COSBN, bias_idx] = -0.5 * k_sharp

    # Offset corridor detectors.
    W[NEAR_E, POS_X] = bump_scale
    Win[NEAR_E, bias_idx] = drift_offset * bump_scale
    W[NEAR_W, POS_X] = bump_scale
    Win[NEAR_W, bias_idx] = -drift_offset * bump_scale

    # NCR_E / NCR_W: sharp ANDs against the bump detectors.
    W[NCR_E, COSBP] =  k_sharp
    W[NCR_E, NEAR_E] = k_sharp
    Win[NCR_E, bias_idx] = -1.2 * k_sharp

    W[NCR_W, COSBN] =  k_sharp
    W[NCR_W, NEAR_W] = k_sharp
    Win[NCR_W, bias_idx] = -1.2 * k_sharp

    # COS_SMALL: |cos| <= 0.5  ≡  NOT cos_big_pos AND NOT cos_big_neg.
    W[COS_SMALL, COSBP] = -k_sharp
    W[COS_SMALL, COSBN] = -k_sharp
    Win[COS_SMALL, bias_idx] = 0.5 * k_sharp

    # NCR_C: COS_SMALL AND NEAR_C (fallback when bot is heading nearly vertical).
    W[NCR_C, COS_SMALL] = k_sharp
    W[NCR_C, NEAR_C] = k_sharp
    Win[NCR_C, bias_idx] = -1.2 * k_sharp

    # NEAR_CR: OR of NCR_E, NCR_W, NCR_C.
    W[NEAR_CR, NCR_E] = k_sharp
    W[NEAR_CR, NCR_W] = k_sharp
    W[NEAR_CR, NCR_C] = k_sharp
    Win[NEAR_CR, bias_idx] = -0.5 * k_sharp

    # TRIG_SC: AND of (reward_latch>0.5, HH, FC, NEAR_CR).
    # Use near_cr (offset-shifted) instead of NEAR_C so the shortcut
    # actually fires when the bot is on-corridor under a slanted heading.
    # Historical AND terms SCI (sc_countdown < 0.5) and ES (old step counter) were removed:
    # SCI was redundant with the -k_sharp * sc_countdown(t) refractory below, and ES
    # had no empirical effect on the accepted score. Bias re-centered
    # to -3.5 * k_sharp for the AND-of-4.
    W[TSC, REWARD_LATCH] = k_sharp
    W[TSC, HH] =  k_sharp
    W[TSC, FC] =  k_sharp
    W[TSC, NEAR_CR] = k_sharp
    Win[TSC, bias_idx] = -k_sharp * 3.5
    # Refractory: once trig_sc fires, keep it suppressed through two paths:
    # W[TSC, TSC] blocks t+1 directly, and W[TSC, SC_COUNTDOWN] blocks
    # the remaining countdown window.
    W[TSC, TSC] = -k_sharp * 10
    W[TSC, SC_COUNTDOWN] = -k_sharp

    # on_countdown: sc_countdown > 0.5.
    W[ONC, SC_COUNTDOWN] = k_sharp
    Win[ONC, bias_idx] = -0.5 * k_sharp

    # is_turn: sc_countdown > approach_steps + 0.5.
    W[IST, SC_COUNTDOWN] = k_sharp
    Win[IST, bias_idx] = -k_sharp * (float(approach_steps) + 0.5)

    # IS_APP: on_countdown AND NOT is_turn.
    W[ISA, ONC] = k_sharp
    W[ISA, IST]  = -k_sharp
    Win[ISA, bias_idx] = -0.5 * k_sharp
    # Turn direction quadrants.
    # turn_toward = -sign(cos) * sign(y).
    # Quadrant ANDs (each is a 3-way AND of COS sign, Y sign, IS_TURN):
    #   cy_pp: cos+, y+  →  -1
    #   cy_pn: cos+, y-  →  +1
    #   cy_np: cos-, y+  →  +1
    #   cy_nn: cos-, y-  →  -1
    # Each CY_* is z = K*(a + b + c - 2.5), rt ~1 iff all three are 1.
    W[CY_PP, COS_POS] = k_sharp
    W[CY_PP, Y_POS]   = k_sharp
    W[CY_PP, IST]     = k_sharp
    Win[CY_PP, bias_idx] = -2.5 * k_sharp

    W[CY_PN, COS_POS] = k_sharp
    W[CY_PN, Y_NEG]   = k_sharp
    W[CY_PN, IST]     = k_sharp
    Win[CY_PN, bias_idx] = -2.5 * k_sharp

    W[CY_NP, COS_NEG] = k_sharp
    W[CY_NP, Y_POS]   = k_sharp
    W[CY_NP, IST]     = k_sharp
    Win[CY_NP, bias_idx] = -2.5 * k_sharp

    W[CY_NN, COS_NEG] = k_sharp
    W[CY_NN, Y_NEG]   = k_sharp
    W[CY_NN, IST]     = k_sharp
    Win[CY_NN, bias_idx] = -2.5 * k_sharp
    # Copy Wout row 0 into W[dtheta, :] as the last step so
    # z_dtheta(t+1) = O(t).
    for j in range(n):
        if Wout[0, j] != 0.0:
            W[DTHETA, j] = Wout[0, j]
    # Build pointwise activation.
    f = make_activation(a, idx)

    model = Win, W, Wout, warmup, leak, f, g
    yield model


if __name__ == "__main__":
    import time
    from challenge_2 import evaluate, train

    seed = 12345
    np.random.seed(seed)
    print("Training reflex_bio2 player for env2 (single yield, instant)...")
    model = train(reflex_bio2_player, timeout=100)

    W_in, W, W_out, warmup, leak, f, g = model

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score (distance): {score:.2f} +/- {std:.2f}")
