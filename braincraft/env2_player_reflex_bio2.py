# Braincraft challenge - Reflex Bio Player 2 for Environment 2
# Self-contained: no dependency on other reflex players.

"""
Reflex Bio Player 2 for Environment 2.

Same behaviour as env2_player_reflex_bio.py but more biologically
plausible:

  * The activation function f(x) is a purely pointwise per-neuron map
    - each out[i] depends only on x[i] and a fixed per-neuron choice
    from a small library of simple scalar functions:

        relu_tanh  : max(0, tanh(z))       (default, thresholding)
        identity   : z                     (accumulators, passthrough)
        relu       : max(0, z)             (counters that grow > 1)
        clip_a     : clip(z, -a, a)        (dtheta output)
        sin        : np.sin(z)             (trig neurons)
        square     : z*z                   (magnitude tests)
        bump       : max(0, 1 - 4*z*z)     (compact support detector)

  * All non-pointwise computations that used to live inside the Python
    activation (gating, latches, state machines, trigonometry, position
    updates, shortcut switching) are expressed with extra helper neurons
    and weights in Win / W. No runtime Python logic influences the
    dynamics - the network is a pure reservoir: X <- f(Win·I + W·X).

The behaviour should be close to the original bio player.  Some circuits
introduce an extra 1–2 step delay compared to the in-f version; this is
negligible given the slow time-scales of wall-following (~1000 steps
per run).

Neuron layout (selected):

  X0..X4    : reflex sensor features (relu_tanh)
  X5        : unused legacy slot
  X6        : dtheta output      (clip to ±a)
  X7        : direction accum.   (identity)
  X8, X9    : unused legacy slots
  X10, X11  : position x / y     (identity)
  X12       : unused legacy slot
  X13, X14  : correction latch / instantaneous (identity)
  X15..X18  : reward circuit     (relu_tanh)
  X19       : unused legacy slot (former step counter)
  X20       : shortcut_steer     (identity)
  X22       : shortcut countdown (relu)
  X23       : init_impulse       (identity)
  X24       : seeded flag        (relu_tanh)
  X25..     : xi_blue (+ xi_red only if kept), then bio helpers

The trig pair keeps the current output semantics while using only the
`sin` activation. This preserves the established corridor logic and the
rotated X10/X11 frame used by the downstream predicates.

The shortcut state machine and the initial-heading correction are
expressed as networks of thresholded rt neurons with sharp gains so
their behaviour approximates the logical ifs of the original.
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_2 import Environment


# ── Shortcut circuit parameters ────────────────────────────────────
SHORTCUT_TURN    = -2.0
SIN_HORIZ_THR    = 0.35
SIN_VERT_THR     = 0.70
COUNTER_THR      = 60
NEAR_CENTER_THR  = 0.05
DRIFT_OFFSET     = 0.175
TURN_STEPS       = 18
APPROACH_STEPS   = 45
SC_TOTAL         = TURN_STEPS + APPROACH_STEPS

# ── Bio-specific constants ─────────────────────────────────────────
COLOR_EVIDENCE_THRESHOLD = 2.0
FRONT_GAIN_MAG   = np.radians(20.0)
GATE_C           = 1.0
USE_XI_RED_FALLBACK = False

# Sharp-threshold gain for crisp AND / OR / latch circuits.
K_SHARP = 50.0     # large enough to saturate tanh near |z|>0.1
STEP_A  = np.radians(5.0)


# ── Neuron index layout ────────────────────────────────────────────
def _bio2_indices(n_rays, use_xi_red=False):
    """Sequential neuron indices for the bio2 layout."""
    xi_blue_start = 25
    xi_blue_stop  = xi_blue_start + n_rays

    i = xi_blue_stop
    idx = {
        "shortcut_steer": 20,
        "init_impulse": 23,
        "xi_blue_start": xi_blue_start,
        "xi_blue_stop": xi_blue_stop,
        "half": n_rays // 2,
    }
    if use_xi_red:
        idx["xi_red_start"] = i
        idx["xi_red_stop"] = i + n_rays
        i = idx["xi_red_stop"]

    def alloc(name):
        nonlocal i
        idx[name] = i
        i += 1

    # Colour evidence & front-sign latches.
    alloc("l_ev")
    alloc("r_ev")
    alloc("dleft")         # rt pulse: left-only AND not latched
    alloc("dright")        # rt pulse: right-only AND not latched
    alloc("evidence")      # signed integrator (identity)
    alloc("trig_pos")      # rt: evidence > +thr
    alloc("trig_neg")      # rt: evidence < -thr
    alloc("fs_pos")        # rt self-latch: front sign positive
    alloc("fs_neg")        # rt self-latch: front sign negative
    alloc("x5_pos")        # rt: gated front block, positive side
    alloc("x5_neg")        # rt: gated front block, negative side

    # Trig neurons.
    alloc("cos_n")         # cos(theta)
    alloc("sin_n")         # sin(theta)

    # Magnitude/sign helpers.
    alloc("sin_sq")        # SIN_N squared
    alloc("cos_pos")       # rt(K*COS_N)
    alloc("cos_neg")       # rt(-K*COS_N)
    alloc("y_pos")         # rt(K*X11)
    alloc("y_neg")         # rt(-K*X11)

    # Shortcut predicates.
    alloc("near_center")   # |X10| < NEAR_CENTER_THR
    alloc("heading_horiz") # sin^2 < SIN_HORIZ_THR^2
    alloc("front_clear")   # X5p + X5n small
    alloc("trig_sc")       # AND of 4 conditions (refractory via -K*X22)

    # Shortcut phase indicators.
    alloc("on_22")         # X22 > 0.5
    alloc("is_turn")       # X22 > APPROACH_STEPS + 0.5
    alloc("is_app")        # on_22 AND NOT is_turn

    # Seeded latch helpers (for capturing initial correction).
    alloc("seed_pos")      # rt: captures +part of (-corr) at step 0
    alloc("seed_neg")      # rt: captures -part

    # Turn-direction quadrant ANDs.
    alloc("cy_pp")         # rt: COS_POS AND Y_POS AND IS_TURN
    alloc("cy_pn")         # rt: COS_POS AND Y_NEG AND IS_TURN
    alloc("cy_np")         # rt: COS_NEG AND Y_POS AND IS_TURN
    alloc("cy_nn")         # rt: COS_NEG AND Y_NEG AND IS_TURN

    # near_corr predicate helpers.
    alloc("cos_big_pos")   # rt: cos > +0.5
    alloc("cos_big_neg")   # rt: cos < -0.5
    alloc("near_e")        # bump: |x10 + DRIFT_OFFSET| < THR
    alloc("near_w")        # bump: |x10 - DRIFT_OFFSET| < THR
    alloc("ncr_e")         # rt: cos_big_pos AND |x10+DRIFT|<THR
    alloc("ncr_w")         # rt: cos_big_neg AND |x10-DRIFT|<THR
    alloc("cos_small")     # rt: |cos| <= 0.5
    alloc("ncr_c")         # rt: cos_small AND near_center (fallback)
    alloc("near_corr")     # rt: OR of ncr_e, ncr_w, ncr_c

    alloc("bio_end")
    return idx


# ── Pointwise activation function ─────────────────────────────────
def make_activation(a, idx):
    """Return the per-neuron pointwise activation.

    Each out[i] depends only on x[i].  The activation type for neuron i
    is looked up from index arrays built once here.
    """
    id_list = [7, 10, 11, 13, 14, idx["shortcut_steer"], idx["init_impulse"], idx["evidence"]]
    sin_list = [idx["cos_n"], idx["sin_n"]]
    square_list = [idx["sin_sq"]]
    relu_list = [22]
    bump_list = [idx["near_center"], idx["near_e"], idx["near_w"]]
    bump_list.extend(range(idx["xi_blue_start"], idx["xi_blue_stop"]))
    if "xi_red_start" in idx:
        bump_list.extend(range(idx["xi_red_start"], idx["xi_red_stop"]))

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
        # X6 : clip(z, -a, a)
        out[6, 0] = float(np.clip(x[6, 0], -a, a))
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
    idx = _bio2_indices(n_rays, use_xi_red=USE_XI_RED_FALLBACK)
    a = STEP_A

    # Aliases.
    L_EV     = idx["l_ev"]
    R_EV     = idx["r_ev"]
    DLEFT    = idx["dleft"]
    DRIGHT   = idx["dright"]
    EV       = idx["evidence"]
    TP       = idx["trig_pos"]
    TN       = idx["trig_neg"]
    FS_P     = idx["fs_pos"]
    FS_N     = idx["fs_neg"]
    X5P      = idx["x5_pos"]
    X5N      = idx["x5_neg"]
    SHORTCUT_STEER = idx["shortcut_steer"]
    INIT_IMPULSE   = idx["init_impulse"]
    COS_N    = idx["cos_n"]
    SIN_N    = idx["sin_n"]
    SIN_SQ   = idx["sin_sq"]
    COS_POS  = idx["cos_pos"]
    COS_NEG  = idx["cos_neg"]
    Y_POS    = idx["y_pos"]
    Y_NEG    = idx["y_neg"]
    NC       = idx["near_center"]
    HH       = idx["heading_horiz"]
    FC       = idx["front_clear"]
    TSC      = idx["trig_sc"]
    ON22     = idx["on_22"]
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
    NEAR_CORR = idx["near_corr"]
    CY_PP    = idx["cy_pp"]
    CY_PN    = idx["cy_pn"]
    CY_NP    = idx["cy_np"]
    CY_NN    = idx["cy_nn"]

    # ── Reflex feature neurons ───────────────────────────────────
    L_idx          = 20
    R_idx          = 63 - L_idx         # 43
    left_side_idx  = 11
    right_side_idx = 63 - left_side_idx # 52

    Win[0, hit_idx]        = 1.0        # X0: hit
    Win[1, L_idx]          = 1.0        # X1: prox[L]
    Win[2, R_idx]          = 1.0        # X2: prox[R]
    Win[3, left_side_idx]  = -1.0       # X3: safety L
    Win[4, right_side_idx] = -1.0       # X4: safety R

    # Approach-phase cancellation: silence reflex features 0..4 while
    # IS_APP is active, so O reduces to front_block + X20 (matches bio's
    # `x20_val = -O_no_front` cancellation during approach phase).
    for i in range(5):
        W[i, ISA] = -K_SHARP * 100.0

    C1_idx, C2_idx = 31, 32
    front_thr      = 1.4

    TANH1 = np.tanh(1.0)

    hit_turn          = np.radians(-10.0) / TANH1
    heading_gain      = np.radians(-40.0)
    safety_gain_left  = np.radians(-20.0)
    safety_gain_right = -safety_gain_left
    safety_target     = 0.75
    Win[3, bias_idx]  = safety_target
    Win[4, bias_idx]  = safety_target

    Wout[0, 0] = hit_turn
    Wout[0, 1] = heading_gain
    Wout[0, 2] = -heading_gain
    Wout[0, 3] = safety_gain_left
    Wout[0, 4] = safety_gain_right
    Wout[0, SHORTCUT_STEER] = 1.0
    Wout[0, INIT_IMPULSE] = 1.0

    # ── X6: dtheta = clip(O, -a, a) ------------------------------
    # z[6] = Wout[0, :] @ X_prev = O_{t-1} (1-step lag).
    # We fill W[6, :] AFTER all Wout entries are finalized (bottom).

    # ── X7: direction accumulator ---------------------------------
    W[7, 7] = 1.0
    W[7, 6] = 1.0

    # ── X10, X11: position accumulators ---------------------------
    # No hit-gating: drift during transient hits is negligible.
    # Under trig option 2A both neurons use the sin activation, but the
    # biases preserve the current downstream frame exactly.
    W[10, 10]   = 1.0
    W[10, COS_N] = speed
    W[11, 11]   = 1.0
    W[11, SIN_N] = speed

    # ── X13 (locked initial heading correction, identity) ------
    # Bio1 latches X[13] = current_corr_0 from step 0 onward via
    # hold_val = x13 - x14 = X_prev[13] (the subtraction cancels the
    # current-step term, so X[13] only carries the initial value).
    # In bio2 we replicate this with a self-recurring latch seeded by
    # SEEDP/SEEDN (which fire only at step 0 before X24 latches).
    # There is a 1-step lag vs bio1 (X[13] = 0 at step 0, current_corr_0
    # from step 1 onward), but the first-step impact is negligible.
    cal_gain = 1.0 / 0.173
    W[13, 13]    =  1.0
    W[13, SEEDP] =  1.0
    W[13, SEEDN] = -1.0

    # X14 (kept for parity with bio layout; not read anywhere).
    Win[14, L_idx] = -cal_gain
    Win[14, R_idx] =  cal_gain

    # ── Reward circuit X15-X18 (unchanged) -----------------------
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

    # ── X24: seeded flag (rt that saturates to 1 at step 1) -----
    Win[24, bias_idx] = 10.0

    # Slot 19 is unused (former step counter, retired with ES removal).

    # ── Shortcut outputs -----------------------------------------
    # Route shortcut steering and the one-shot init impulse through
    # separate fixed slots so debug/docs can name them directly.
    W[SHORTCUT_STEER, CY_PN] =  abs(SHORTCUT_TURN)   # cos+, y-  → +2
    W[SHORTCUT_STEER, CY_NP] =  abs(SHORTCUT_TURN)   # cos-, y+  → +2
    W[SHORTCUT_STEER, CY_PP] = -abs(SHORTCUT_TURN)   # cos+, y+  → -2
    W[SHORTCUT_STEER, CY_NN] = -abs(SHORTCUT_TURN)   # cos-, y-  → -2

    # One-shot init correction: SEED_{POS,NEG} fire only before X24
    # latches, so INIT_IMPULSE contributes -current_corr for one step.
    W[INIT_IMPULSE, SEEDP] = -1.0
    W[INIT_IMPULSE, SEEDN] =  1.0

    # ── X22: shortcut countdown (relu) ---------------------------
    # z[22] = X_prev[22] - 1 + (SC_TOTAL+1)·TSC_prev
    W[22, 22] = 1.0
    Win[22, bias_idx] = -1.0
    W[22, TSC] = float(SC_TOTAL) + 1.0

    # SEED_POS / SEED_NEG: capture +/- part of current_corr ONLY when
    # X24_prev is 0 (unseeded).  After X24 latches to 1 at step 1+,
    # a huge negative gate suppresses them.
    # SEED_POS fires when current_corr > 0 AND not yet seeded.
    Win[SEEDP, L_idx] = -cal_gain
    Win[SEEDP, R_idx] =  cal_gain
    W[SEEDP, 24] = -1.0e3        # when X24=1, z ≈ -1000 → rt = 0

    Win[SEEDN, L_idx] =  cal_gain
    Win[SEEDN, R_idx] = -cal_gain
    W[SEEDN, 24] = -1.0e3

    # ── Colour evidence rays (bump-based Xi_blue, optional Xi_red) --
    for r in range(n_rays):
        color_input_col = p + r
        Win[idx["xi_blue_start"] + r, color_input_col] = 1.0
        Win[idx["xi_blue_start"] + r, bias_idx] = -4.0
        if "xi_red_start" in idx:
            Win[idx["xi_red_start"] + r, color_input_col] = 1.0
            Win[idx["xi_red_start"] + r, bias_idx] = -5.0

    half = idx["half"]
    for r in range(half):
        W[L_EV, idx["xi_blue_start"] + r] = 1.0
        if "xi_red_start" in idx:
            W[L_EV, idx["xi_red_start"] + r] = -3.0
    for r in range(half, n_rays):
        W[R_EV, idx["xi_blue_start"] + r] = 1.0
        if "xi_red_start" in idx:
            W[R_EV, idx["xi_red_start"] + r] = -3.0

    # ── Gated delta pulses (DLEFT / DRIGHT) ---------------------
    # Fire only when one side dominates AND front sign NOT yet latched.
    # With K_SHARP=50 gain and rt, saturates to ~1 when L/R difference
    # exceeds a margin.  The latch subtraction zeros them out once
    # FS_POS or FS_NEG is 1.
    W[DLEFT, L_EV]  =  K_SHARP
    W[DLEFT, R_EV]  = -K_SHARP
    W[DLEFT, FS_P]  = -K_SHARP * 10
    W[DLEFT, FS_N]  = -K_SHARP * 10
    Win[DLEFT, bias_idx] = -0.2 * K_SHARP

    W[DRIGHT, L_EV] = -K_SHARP
    W[DRIGHT, R_EV] =  K_SHARP
    W[DRIGHT, FS_P] = -K_SHARP * 10
    W[DRIGHT, FS_N] = -K_SHARP * 10
    Win[DRIGHT, bias_idx] = -0.2 * K_SHARP

    # ── EVIDENCE (signed integrator, identity) ------------------
    W[EV, EV]     = 1.0
    W[EV, DRIGHT] = 1.0
    W[EV, DLEFT]  = -1.0

    # ── Trigger thresholds (rt, sharp) --------------------------
    # bio1 uses `if evidence_now >= threshold`; rt(K*(EV-thr)) for EV==thr
    # gives rt(0)=0, so we lower the threshold by 0.5 so TP fires at
    # EV==threshold (DLEFT/DRIGHT saturate to ~±1, so EV is integer-ish).
    W[TP, EV] =  K_SHARP
    Win[TP, bias_idx] = -K_SHARP * (COLOR_EVIDENCE_THRESHOLD - 0.5)
    W[TN, EV] = -K_SHARP
    Win[TN, bias_idx] = -K_SHARP * (COLOR_EVIDENCE_THRESHOLD - 0.5)

    # ── FS latches (rt self-latching) ---------------------------
    W[FS_P, FS_P] = K_SHARP
    W[FS_P, TP]   = K_SHARP
    W[FS_N, FS_N] = K_SHARP
    W[FS_N, TN]   = K_SHARP

    # ── X5_POS / X5_NEG (gated front block) ---------------------
    Win[X5P, C1_idx]   = 1.0
    Win[X5P, C2_idx]   = 1.0
    Win[X5P, bias_idx] = -(front_thr + GATE_C)
    W[X5P, FS_P]     =  GATE_C
    W[X5P, FS_N]     = -GATE_C

    Win[X5N, C1_idx]   = 1.0
    Win[X5N, C2_idx]   = 1.0
    Win[X5N, bias_idx] = -(front_thr + GATE_C)
    W[X5N, FS_P]     = -GATE_C
    W[X5N, FS_N]     =  GATE_C

    Wout[0, X5P] =  FRONT_GAIN_MAG
    Wout[0, X5N] = -FRONT_GAIN_MAG

    # ── Trig neurons (2A: sin-only activation, preserved outputs) --
    for idx_trig in (COS_N, SIN_N):
        W[idx_trig, 7] = 1.0
        W[idx_trig, 13] = 1.0
        W[idx_trig, 6] = 1.0
    Win[COS_N, bias_idx] = np.pi
    Win[SIN_N, bias_idx] = np.pi / 2

    # ── Magnitude/sign helpers ----------------------------------
    W[SIN_SQ, SIN_N] = 1.0

    # COS_POS/COS_NEG only feed quadrant sign logic, so saturation is fine.
    W[COS_POS, COS_N] =  K_SHARP
    W[COS_NEG, COS_N] = -K_SHARP

    # Y_POS/Y_NEG feed quadrant sign logic as sign(y) - saturation OK.
    W[Y_POS,   11]    =  K_SHARP
    W[Y_NEG,   11]    = -K_SHARP

    # Bump support is |z| < 0.5, so use z = value / (2*thr).
    bump_scale = 1.0 / (2.0 * NEAR_CENTER_THR)
    W[NC, 10] = bump_scale

    # HH reads the exact squared sine magnitude.
    W[HH, SIN_SQ] = -K_SHARP / (SIN_HORIZ_THR ** 2)
    Win[HH, bias_idx] = K_SHARP

    # FRONT_CLEAR: X5p + X5n < 0.1.
    W[FC, X5P] = -K_SHARP
    W[FC, X5N] = -K_SHARP
    Win[FC, bias_idx] =  K_SHARP * 0.1

    # ── near_corr helpers ───────────────────────────────────────
    # COS_BIG_POS: cos > 0.5 (sharp).
    W[COSBP, COS_N] =  K_SHARP
    Win[COSBP, bias_idx] = -0.5 * K_SHARP
    # COS_BIG_NEG: cos < -0.5.
    W[COSBN, COS_N] = -K_SHARP
    Win[COSBN, bias_idx] = -0.5 * K_SHARP

    # Offset corridor detectors.
    W[NEAR_E, 10] = bump_scale
    Win[NEAR_E, bias_idx] = DRIFT_OFFSET * bump_scale
    W[NEAR_W, 10] = bump_scale
    Win[NEAR_W, bias_idx] = -DRIFT_OFFSET * bump_scale

    # NCR_E / NCR_W: sharp ANDs against the bump detectors.
    W[NCR_E, COSBP] =  K_SHARP
    W[NCR_E, NEAR_E] = K_SHARP
    Win[NCR_E, bias_idx] = -1.2 * K_SHARP

    W[NCR_W, COSBN] =  K_SHARP
    W[NCR_W, NEAR_W] = K_SHARP
    Win[NCR_W, bias_idx] = -1.2 * K_SHARP

    # COS_SMALL: |cos| <= 0.5  ≡  NOT cos_big_pos AND NOT cos_big_neg.
    W[COS_SMALL, COSBP] = -K_SHARP
    W[COS_SMALL, COSBN] = -K_SHARP
    Win[COS_SMALL, bias_idx] = 0.5 * K_SHARP

    # NCR_C: COS_SMALL AND NC  (fallback when bot is heading nearly vertical).
    W[NCR_C, COS_SMALL] = K_SHARP
    W[NCR_C, NC]        = K_SHARP
    Win[NCR_C, bias_idx] = -1.2 * K_SHARP

    # NEAR_CORR: OR of NCR_E, NCR_W, NCR_C.
    W[NEAR_CORR, NCR_E] = K_SHARP
    W[NEAR_CORR, NCR_W] = K_SHARP
    W[NEAR_CORR, NCR_C] = K_SHARP
    Win[NEAR_CORR, bias_idx] = -0.5 * K_SHARP

    # TRIG_SC: AND of (X18>0.5, HH, FC, NEAR_CORR).
    # Use near_corr (offset-shifted) instead of NC so the shortcut
    # actually fires when the bot is on-corridor under a slanted heading.
    # Historical AND terms SCI (X22<0.5) and ES (X19>60) were removed:
    # SCI was redundant with the `-K·X22_prev` refractory below, and ES
    # had no empirical effect on the accepted score. Bias re-centered
    # to -3.5·K for the AND-of-4.
    W[TSC, 18] =  K_SHARP
    W[TSC, HH] =  K_SHARP
    W[TSC, FC] =  K_SHARP
    W[TSC, NEAR_CORR] =  K_SHARP
    Win[TSC, bias_idx] = -K_SHARP * 3.5
    # Refractory: once TSC fires, keep it suppressed through two paths:
    #   • W[TSC, TSC]: blocks t+1 (reads TSC_prev directly)
    #   • W[TSC, 22]:  blocks t+2..t+SC_TOTAL (reads X22 countdown;
    #     at X22_prev>=1 this contributes <=-K, which together with the
    #     -4.5·K bias sinks the AND below zero even if all 5 positive
    #     inputs are saturated to 1)
    W[TSC, TSC] = -K_SHARP * 10
    W[TSC, 22]  = -K_SHARP

    # ON_22: X22 > 0.5.
    W[ON22, 22] = K_SHARP
    Win[ON22, bias_idx] = -0.5 * K_SHARP

    # IS_TURN: X22 > APPROACH_STEPS + 0.5.
    W[IST, 22] = K_SHARP
    Win[IST, bias_idx] = -K_SHARP * (float(APPROACH_STEPS) + 0.5)

    # IS_APP: on_22 AND NOT is_turn.
    W[ISA, ON22] =  K_SHARP
    W[ISA, IST]  = -K_SHARP
    Win[ISA, bias_idx] = -0.5 * K_SHARP

    # ── Turn direction quadrants ----------------------------------
    # turn_toward = -sign(cos) * sign(y).
    # Quadrant ANDs (each is a 3-way AND of COS sign, Y sign, IS_TURN):
    #   cy_pp: cos+, y+  →  -1
    #   cy_pn: cos+, y-  →  +1
    #   cy_np: cos-, y+  →  +1
    #   cy_nn: cos-, y-  →  -1
    # Each CY_* is z = K*(a + b + c - 2.5), rt ~1 iff all three are 1.
    W[CY_PP, COS_POS] = K_SHARP
    W[CY_PP, Y_POS]   = K_SHARP
    W[CY_PP, IST]     = K_SHARP
    Win[CY_PP, bias_idx] = -2.5 * K_SHARP

    W[CY_PN, COS_POS] = K_SHARP
    W[CY_PN, Y_NEG]   = K_SHARP
    W[CY_PN, IST]     = K_SHARP
    Win[CY_PN, bias_idx] = -2.5 * K_SHARP

    W[CY_NP, COS_NEG] = K_SHARP
    W[CY_NP, Y_POS]   = K_SHARP
    W[CY_NP, IST]     = K_SHARP
    Win[CY_NP, bias_idx] = -2.5 * K_SHARP

    W[CY_NN, COS_NEG] = K_SHARP
    W[CY_NN, Y_NEG]   = K_SHARP
    W[CY_NN, IST]     = K_SHARP
    Win[CY_NN, bias_idx] = -2.5 * K_SHARP

    # ── X6 = clip(Wout @ X_prev) --------------------------------
    # Copy Wout row 0 into W row 6 as the last step so z[6] = O_{t-1}.
    for j in range(n):
        if Wout[0, j] != 0.0:
            W[6, j] = Wout[0, j]

    # ── Build activation ---------------------------------------
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
