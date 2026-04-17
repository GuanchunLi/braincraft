# Braincraft challenge - Reflex Bio Player 2 for Environment 2
# Self-contained: no dependency on other reflex players.

"""
Reflex Bio Player 2 for Environment 2.

Same behaviour as env2_player_reflex_bio.py but more biologically
plausible:

  * The activation function f(x) is a purely pointwise per-neuron map
    — each out[i] depends only on x[i] and a fixed per-neuron choice
    from a small library of simple scalar functions:

        relu_tanh  : max(0, tanh(z))       (default, thresholding)
        identity   : z                     (accumulators, passthrough)
        relu       : max(0, z)             (counters that grow > 1)
        clip_a     : clip(z, -a, a)        (dtheta output)
        sin        : np.sin(z)             (sin neuron)
        cos        : np.cos(z)             (cos neuron)

  * All non-pointwise computations that used to live inside the Python
    activation (gating, latches, state machines, trigonometry, position
    updates, shortcut switching, initial-heading-correction depletion)
    are expressed with extra helper neurons and weights in Win / W.
    No runtime Python logic influences the dynamics — the network is a
    pure reservoir: X ← f(Win·I + W·X).

The behaviour should be close to the original bio player.  Some circuits
introduce an extra 1–2 step delay compared to the in-f version; this is
negligible given the slow time-scales of wall-following (~1000 steps
per run).

Neuron layout (selected):

  X0..X4    : reflex sensor features (relu_tanh)
  X5        : front-block, original — not used in Wout, kept at 0
  X6        : dtheta output      (clip to ±a)
  X7        : direction accum.   (identity)
  X8, X9    : cos/sin            (cos, sin)
  X10, X11  : position x / y     (identity)
  X12       : hit copy           (relu_tanh)
  X13, X14  : correction latch / instantaneous (identity)
  X15..X18  : reward circuit     (relu_tanh)
  X19       : step counter       (relu)
  X20       : shortcut steering  (identity)
  X22       : shortcut countdown (relu)
  X23       : init correction rem (identity)
  X24       : seeded flag        (relu_tanh)
  X25..X152 : Xi_hi / Xi_lo
  X153+     : bio helpers (L_ev, R_ev, TRIG, FS latches, X5p/X5n,
              trig abs-part helpers, discrete shortcut predicates,
              init-correction ramps, etc.)

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

# ── Initial heading correction parameters ──────────────────────────
INIT_CORR_EPS    = 1e-3
INIT_CORR_CAP    = 0.15

# ── Bio-specific constants ─────────────────────────────────────────
COLOR_EVIDENCE_THRESHOLD = 2.0
FRONT_GAIN_MAG   = np.radians(20.0)
BLUE_HI_THR      = 3.5
BLUE_LO_THR      = 4.5
XI_LO_FACTOR     = 3.0
GATE_C           = 1.0

# Sharp-threshold gain for crisp AND / OR / latch circuits.
K_SHARP = 50.0     # large enough to saturate tanh near |z|>0.1
STEP_A  = np.radians(5.0)


# ── Neuron index layout ────────────────────────────────────────────
def _bio2_indices(n_rays):
    """Sequential neuron indices for the bio2 layout."""
    xi_hi_start = 25
    xi_hi_stop  = xi_hi_start + n_rays
    xi_lo_start = xi_hi_stop
    xi_lo_stop  = xi_lo_start + n_rays

    i = xi_lo_stop
    idx = {
        "xi_hi_start": xi_hi_start, "xi_hi_stop": xi_hi_stop,
        "xi_lo_start": xi_lo_start, "xi_lo_stop": xi_lo_stop,
        "half": n_rays // 2,
    }

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

    # Absolute-value helpers (rt).
    alloc("sin_pos")       # rt(K*SIN_N)
    alloc("sin_neg")       # rt(-K*SIN_N)
    alloc("cos_pos")       # rt(K*COS_N)
    alloc("cos_neg")       # rt(-K*COS_N)
    alloc("y_pos")         # rt(K*X11)
    alloc("y_neg")         # rt(-K*X11)
    alloc("x_pos")         # rt(K*X10)
    alloc("x_neg")         # rt(-K*X10)

    # Shortcut predicates (rt, binary-ish).
    alloc("near_center")   # |X10| < NEAR_CENTER_THR
    alloc("heading_vert")  # |sin| > SIN_VERT_THR
    alloc("heading_horiz") # |sin| < SIN_HORIZ_THR
    alloc("front_clear")   # X5p + X5n small
    alloc("enough_steps")  # X19 > COUNTER_THR
    alloc("sc_idle")       # X22 < 0.5
    alloc("nc_and_hv")     # near_center AND heading_vert  (for counter reset)
    alloc("trig_sc")       # AND of 6 conditions

    # Shortcut phase indicators.
    alloc("on_22")         # X22 > 0.5
    alloc("is_turn")       # X22 > APPROACH_STEPS + 0.5
    alloc("is_app")        # on_22 AND NOT is_turn

    # Turn direction (cos·y sign decomposition, gated by IS_TURN).
    alloc("tt_plus")       # (-cos AND +y) OR (+cos AND -y), AND IS_TURN
    alloc("tt_minus")      # (+cos AND +y) OR (-cos AND -y), AND IS_TURN

    # Init correction depletion branches.
    alloc("x23p")          # rt: sign(X23) positive AND not small
    alloc("x23n")          # rt: sign(X23) negative AND not small
    alloc("is_init")       # |X23| > eps AND X24 latched

    # Seeded latch helpers (for capturing initial correction).
    alloc("seed_pos")      # rt: captures +part of (-corr) at step 0
    alloc("seed_neg")      # rt: captures -part

    # Turn-direction quadrant ANDs (replace fragile TTP/TTM formula).
    alloc("cy_pp")         # rt: COS_POS AND Y_POS AND IS_TURN
    alloc("cy_pn")         # rt: COS_POS AND Y_NEG AND IS_TURN
    alloc("cy_np")         # rt: COS_NEG AND Y_POS AND IS_TURN
    alloc("cy_nn")         # rt: COS_NEG AND Y_NEG AND IS_TURN

    # near_corr predicate helpers (offset-shifted center check).
    alloc("cos_big_pos")   # rt: cos > +0.5
    alloc("cos_big_neg")   # rt: cos < -0.5
    alloc("xe_pos")        # rt: x10 + DRIFT_OFFSET > +NEAR_CENTER_THR
    alloc("xe_neg")        # rt: x10 + DRIFT_OFFSET < -NEAR_CENTER_THR
    alloc("xw_pos")        # rt: x10 - DRIFT_OFFSET > +NEAR_CENTER_THR
    alloc("xw_neg")        # rt: x10 - DRIFT_OFFSET < -NEAR_CENTER_THR
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
    id_list   = [7, 10, 11, 13, 14, 20, 23, idx["evidence"]]
    cos_list  = [idx["cos_n"]]
    sin_list  = [idx["sin_n"]]
    relu_list = [19, 22]

    id_arr   = np.array(sorted(set(id_list)), dtype=int)
    cos_arr  = np.array(sorted(set(cos_list)), dtype=int)
    sin_arr  = np.array(sorted(set(sin_list)), dtype=int)
    relu_arr = np.array(sorted(set(relu_list)), dtype=int)

    def f(x):
        # Default: relu_tanh.
        out = np.maximum(0.0, np.tanh(x))
        if id_arr.size:
            out[id_arr, 0] = x[id_arr, 0]
        if cos_arr.size:
            out[cos_arr, 0] = np.cos(x[cos_arr, 0])
        if sin_arr.size:
            out[sin_arr, 0] = np.sin(x[sin_arr, 0])
        if relu_arr.size:
            out[relu_arr, 0] = np.maximum(0.0, x[relu_arr, 0])
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
    idx = _bio2_indices(n_rays)
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
    COS_N    = idx["cos_n"]
    SIN_N    = idx["sin_n"]
    SIN_POS  = idx["sin_pos"]
    SIN_NEG  = idx["sin_neg"]
    COS_POS  = idx["cos_pos"]
    COS_NEG  = idx["cos_neg"]
    Y_POS    = idx["y_pos"]
    Y_NEG    = idx["y_neg"]
    X_POS    = idx["x_pos"]
    X_NEG    = idx["x_neg"]
    NC       = idx["near_center"]
    HV       = idx["heading_vert"]
    HH       = idx["heading_horiz"]
    FC       = idx["front_clear"]
    ES       = idx["enough_steps"]
    SCI      = idx["sc_idle"]
    NCV      = idx["nc_and_hv"]
    TSC      = idx["trig_sc"]
    ON22     = idx["on_22"]
    IST      = idx["is_turn"]
    ISA      = idx["is_app"]
    TTP      = idx["tt_plus"]
    TTM      = idx["tt_minus"]
    X23P     = idx["x23p"]
    X23N     = idx["x23n"]
    ISI      = idx["is_init"]
    SEEDP    = idx["seed_pos"]
    SEEDN    = idx["seed_neg"]
    COSBP    = idx["cos_big_pos"]
    COSBN    = idx["cos_big_neg"]
    XE_P     = idx["xe_pos"]
    XE_N     = idx["xe_neg"]
    XW_P     = idx["xw_pos"]
    XW_N     = idx["xw_neg"]
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
    Win[5, C1_idx]   = 1.0
    Win[5, C2_idx]   = 1.0
    Win[5, bias_idx] = -front_thr       # X5 kept for diagnostics only

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
    Wout[0, 5] = 0.0
    Wout[0, 20] = 1.0                  # shortcut override channel

    # ── X6: dtheta = clip(O, -a, a) ------------------------------
    # z[6] = Wout[0, :] @ X_prev = O_{t-1} (1-step lag).
    # We fill W[6, :] AFTER all Wout entries are finalized (bottom).

    # ── X7: direction accumulator ---------------------------------
    W[7, 7] = 1.0
    W[7, 6] = 1.0

    # ── X10, X11: position accumulators ---------------------------
    # No hit-gating: drift during transient hits is negligible.
    W[10, 10]   = 1.0
    W[10, COS_N] = speed
    W[11, 11]   = 1.0
    W[11, SIN_N] = speed

    # ── X12: hit copy ---------------------------------------------
    Win[12, hit_idx] = 1.0

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

    # ── X19: step counter (relu) ---------------------------------
    # z[19] = X_prev[19] + 1 - BIG * NCV_prev    (reset when NC AND HV)
    W[19, 19] = 1.0
    Win[19, bias_idx] = 1.0
    W[19, NCV] = -float(1.0e6)   # huge negative pulse resets to 0

    # ── X20: shortcut steering -----------------------------------
    # X20 = (turn_toward·|SHORTCUT_TURN| when IS_TURN)
    #     + (init_dtheta, one-shot at step 2, from SEED_POS/SEED_NEG)
    # Identity activation.  Route cy_* quadrant gates directly into X20
    # (skip TTP/TTM OR intermediate) to reduce the trigger→action lag.
    W[20, CY_PN] =  abs(SHORTCUT_TURN)   # cos+, y-  → +2
    W[20, CY_NP] =  abs(SHORTCUT_TURN)   # cos-, y+  → +2
    W[20, CY_PP] = -abs(SHORTCUT_TURN)   # cos+, y+  → -2
    W[20, CY_NN] = -abs(SHORTCUT_TURN)   # cos-, y-  → -2
    # One-shot init correction: SEED_{POS,NEG} fire ONLY at step 1 (before
    # X24 latches), so the next-step X20 receives -current_corr for exactly
    # one step.  This is the bio2 analogue of bio's X23 integrator+depletion
    # without needing a state variable.
    W[20, SEEDP] = -1.0
    W[20, SEEDN] =  1.0

    # ── X22: shortcut countdown (relu) ---------------------------
    # z[22] = X_prev[22] - 1 + (SC_TOTAL+1)·TSC_prev
    W[22, 22] = 1.0
    Win[22, bias_idx] = -1.0
    W[22, TSC] = float(SC_TOTAL) + 1.0

    # ── X23: initial heading correction remainder -----------------
    # Captured at step 0 via SEED_POS/SEED_NEG, which only fire when
    # NOT X24_prev.  After that, X23 self-recurs and depletes by ±a.
    W[23, 23] = 1.0
    W[23, SEEDP]  = -1.0   # seed = -correction (captured at step 0)
    W[23, SEEDN]  =  1.0
    W[23, X23P]   = -a     # deplete: subtract +a while X23 > eps
    W[23, X23N]   =  a     # deplete: add +a while X23 < -eps

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

    # Note SEED_POS output = rt(current_corr - 1000*X_prev[24]).  With
    # current_corr ~ 0.07 and X_prev[24] = 0 at step 0, output ≈
    # tanh(0.07) ≈ 0.07.  At step 1+, X_prev[24] ≈ 1, output ≈ 0.
    # X23 is integrator with W[23, SEEDP] = -1, so step 0 contributes
    # -tanh(current_corr).  For small corrections, tanh(c) ≈ c, so
    # X23 gets seeded with ≈ -current_corr_0 = same sign as bio's
    # init_remaining = clip(-correction, ±CAP).

    # ── Colour evidence rays (Xi_hi / Xi_lo, same as bio) -------
    for r in range(n_rays):
        color_input_col = p + r
        Win[idx["xi_hi_start"] + r, color_input_col] = 1.0
        Win[idx["xi_hi_start"] + r, bias_idx]        = -BLUE_HI_THR
        Win[idx["xi_lo_start"] + r, color_input_col] = 1.0
        Win[idx["xi_lo_start"] + r, bias_idx]        = -BLUE_LO_THR

    half = idx["half"]
    for r in range(half):
        W[L_EV, idx["xi_hi_start"] + r] =  1.0
        W[L_EV, idx["xi_lo_start"] + r] = -XI_LO_FACTOR
    for r in range(half, n_rays):
        W[R_EV, idx["xi_hi_start"] + r] =  1.0
        W[R_EV, idx["xi_lo_start"] + r] = -XI_LO_FACTOR

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

    # ── Trig neurons (pre-activation = X7 + X13 + X6 + π/2) -----
    for idx_trig in (COS_N, SIN_N):
        W[idx_trig, 7]   = 1.0
        W[idx_trig, 13]  = 1.0
        W[idx_trig, 6]   = 1.0
        Win[idx_trig, bias_idx] = np.pi / 2

    # ── Absolute-value helpers (rt) ----------------------------
    # SIN_POS/SIN_NEG use low gain so rt(|sin|) tracks |sin| smoothly
    # over the full [0, 1] range (no premature saturation).
    BIG_SC = 1.0
    W[SIN_POS, SIN_N] =  BIG_SC
    W[SIN_NEG, SIN_N] = -BIG_SC

    # COS_POS/COS_NEG only feed the TTP/TTM sign logic, where any
    # nonzero cos should pick a side — saturation is fine.
    W[COS_POS, COS_N] =  K_SHARP
    W[COS_NEG, COS_N] = -K_SHARP

    # Y_POS/Y_NEG feed TTP/TTM as sign(y) — saturation OK.
    W[Y_POS,   11]    =  K_SHARP
    W[Y_NEG,   11]    = -K_SHARP

    # X_POS/X_NEG must be binary 0/1 based on |x10| vs NEAR_CENTER_THR.
    # Shifted threshold: fires only when x10 > +THR (resp. x10 < -THR).
    K_POS = K_SHARP / NEAR_CENTER_THR
    W[X_POS,   10]    =  K_POS
    Win[X_POS, bias_idx] = -K_SHARP                     # threshold at +THR
    W[X_NEG,   10]    = -K_POS
    Win[X_NEG, bias_idx] = -K_SHARP                     # threshold at -THR

    # ── NEAR_CENTER: fires when |X10| < NEAR_CENTER_THR -------
    # X_POS, X_NEG both saturate to 1 when |X10| > THR; both stay at 0
    # when |X10| < THR.  NC = NOT(X_POS) AND NOT(X_NEG).
    W[NC, X_POS] = -K_SHARP
    W[NC, X_NEG] = -K_SHARP
    Win[NC, bias_idx] = 0.5 * K_SHARP

    # HV / HH work with the low-gain |sin| signal above.
    # SIN_POS + SIN_NEG ≈ |tanh(sin)| ∈ [0, tanh(1)].
    thr_v = np.tanh(BIG_SC * SIN_VERT_THR)     # |sin|=0.7 → ≈0.604
    thr_h = np.tanh(BIG_SC * SIN_HORIZ_THR)    # |sin|=0.35 → ≈0.336

    # HV: fires when sum > thr_v.
    W[HV, SIN_POS] =  K_SHARP / thr_v
    W[HV, SIN_NEG] =  K_SHARP / thr_v
    Win[HV, bias_idx] = -K_SHARP

    # HH: fires when sum < thr_h.
    W[HH, SIN_POS] = -K_SHARP / thr_h
    W[HH, SIN_NEG] = -K_SHARP / thr_h
    Win[HH, bias_idx] =  K_SHARP

    # FRONT_CLEAR: X5p + X5n < 0.1.
    W[FC, X5P] = -K_SHARP
    W[FC, X5N] = -K_SHARP
    Win[FC, bias_idx] =  K_SHARP * 0.1

    # ENOUGH_STEPS: X19 > COUNTER_THR.
    # X19 is a raw counter; relu-activated, so its value is an int ≥ 0.
    W[ES, 19] = K_SHARP / float(COUNTER_THR)
    Win[ES, bias_idx] = -K_SHARP

    # SC_IDLE: X22 < 0.5.
    W[SCI, 22] = -K_SHARP
    Win[SCI, bias_idx] = 0.5 * K_SHARP

    # NCV: NEAR_CENTER AND HEADING_VERT.  AND gate with two rt inputs
    # each in [0, 1].  Threshold at 1.5.
    W[NCV, NC] =  K_SHARP
    W[NCV, HV] =  K_SHARP
    Win[NCV, bias_idx] = -K_SHARP * 1.5

    # ── near_corr helpers ───────────────────────────────────────
    # COS_BIG_POS: cos > 0.5 (sharp).
    W[COSBP, COS_N] =  K_SHARP
    Win[COSBP, bias_idx] = -0.5 * K_SHARP
    # COS_BIG_NEG: cos < -0.5.
    W[COSBN, COS_N] = -K_SHARP
    Win[COSBN, bias_idx] = -0.5 * K_SHARP

    # XE_{POS,NEG}: shifted to threshold at (x10 + DRIFT) = ±NEAR_CENTER_THR.
    # XE_P fires when x10 > -DRIFT + THR (i.e., east-band upper edge).
    W[XE_P, 10] =  K_POS
    Win[XE_P, bias_idx] =  K_POS * (DRIFT_OFFSET - NEAR_CENTER_THR)
    # XE_N fires when x10 < -DRIFT - THR.
    W[XE_N, 10] = -K_POS
    Win[XE_N, bias_idx] = -K_POS * (DRIFT_OFFSET + NEAR_CENTER_THR)

    # XW_{POS,NEG}: threshold at (x10 - DRIFT) = ±NEAR_CENTER_THR.
    W[XW_P, 10] =  K_POS
    Win[XW_P, bias_idx] = -K_POS * (DRIFT_OFFSET + NEAR_CENTER_THR)
    W[XW_N, 10] = -K_POS
    Win[XW_N, bias_idx] =  K_POS * (DRIFT_OFFSET - NEAR_CENTER_THR)

    # NCR_E: COS_BIG_POS AND |x10+DRIFT|<THR ≡ cos_big_pos AND NOT xe_pos AND NOT xe_neg
    # Preact: K*cos_big_pos - K*xe_pos - K*xe_neg - 0.5*K
    # Fires (to ~1) when cos_big_pos=1 AND xe_pos=0 AND xe_neg=0.
    W[NCR_E, COSBP] =  K_SHARP
    W[NCR_E, XE_P]  = -K_SHARP
    W[NCR_E, XE_N]  = -K_SHARP
    Win[NCR_E, bias_idx] = -0.5 * K_SHARP

    W[NCR_W, COSBN] =  K_SHARP
    W[NCR_W, XW_P]  = -K_SHARP
    W[NCR_W, XW_N]  = -K_SHARP
    Win[NCR_W, bias_idx] = -0.5 * K_SHARP

    # COS_SMALL: |cos| <= 0.5  ≡  NOT cos_big_pos AND NOT cos_big_neg.
    W[COS_SMALL, COSBP] = -K_SHARP
    W[COS_SMALL, COSBN] = -K_SHARP
    Win[COS_SMALL, bias_idx] = 0.5 * K_SHARP

    # NCR_C: COS_SMALL AND NC  (fallback when bot is heading nearly vertical).
    W[NCR_C, COS_SMALL] = K_SHARP
    W[NCR_C, NC]        = K_SHARP
    Win[NCR_C, bias_idx] = -1.5 * K_SHARP

    # NEAR_CORR: OR of NCR_E, NCR_W, NCR_C.
    W[NEAR_CORR, NCR_E] = K_SHARP
    W[NEAR_CORR, NCR_W] = K_SHARP
    W[NEAR_CORR, NCR_C] = K_SHARP
    Win[NEAR_CORR, bias_idx] = -0.5 * K_SHARP

    # TRIG_SC: AND of (X18>0.5, HH, FC, ES, NEAR_CORR, SCI).
    # Use near_corr (offset-shifted) instead of NC so the shortcut
    # actually fires when the bot is on-corridor under a slanted heading.
    W[TSC, 18] =  K_SHARP
    W[TSC, HH] =  K_SHARP
    W[TSC, FC] =  K_SHARP
    W[TSC, ES] =  K_SHARP
    W[TSC, NEAR_CORR] =  K_SHARP
    W[TSC, SCI] = K_SHARP
    Win[TSC, bias_idx] = -K_SHARP * 5.5
    # Refractory: once TSC fires, keep it suppressed.  SCI has a 2-step
    # lag relative to TSC (TSC → X22 → SCI), so we need two independent
    # inhibition paths to avoid TSC firing 3 consecutive times:
    #   • W[TSC, TSC]: blocks t+1 (reads TSC_prev directly)
    #   • W[TSC, 22]:  blocks t+2..t+SC_TOTAL (reads X22 countdown)
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

    # ── Turn direction (TTP / TTM) --------------------------------
    # turn_toward = -sign(cos) * sign(y).
    # Quadrant ANDs (each is a 3-way AND of COS sign, Y sign, IS_TURN):
    #   cy_pp: cos+, y+  →  -1   (TTM)
    #   cy_pn: cos+, y-  →  +1   (TTP)
    #   cy_np: cos-, y+  →  +1   (TTP)
    #   cy_nn: cos-, y-  →  -1   (TTM)
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

    # TTP = OR(cy_pn, cy_np); TTM = OR(cy_pp, cy_nn).
    W[TTP, CY_PN] = K_SHARP
    W[TTP, CY_NP] = K_SHARP
    Win[TTP, bias_idx] = -0.5 * K_SHARP

    W[TTM, CY_PP] = K_SHARP
    W[TTM, CY_NN] = K_SHARP
    Win[TTM, bias_idx] = -0.5 * K_SHARP

    # ── X23 depletion branches (rt) -----------------------------
    # X23P fires when X23 > eps; X23N fires when X23 < -eps.
    # Use BIG_INIT = 1/eps gain so rt saturates for |X23| > eps.
    BIG_INIT = 1.0 / INIT_CORR_EPS
    W[X23P, 23] =  BIG_INIT
    W[X23N, 23] = -BIG_INIT

    # ── IS_INIT: |X23| > eps  AND  X24 latched ------------------
    # We approximate |X23| > eps via X23P + X23N saturation.
    W[ISI, X23P] = K_SHARP
    W[ISI, X23N] = K_SHARP
    W[ISI, 24]   = K_SHARP
    Win[ISI, bias_idx] = -K_SHARP * 1.5

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
