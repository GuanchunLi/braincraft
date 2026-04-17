# Reflex Bio Player 2 - Environment 2

## 1. Overview

[env2_player_reflex_bio2.py](env2_player_reflex_bio2.py) is the
**pointwise-activation** version of the env2 bio controller. It preserves the
behaviour of the earlier `env2_player_reflex_bio.py` but replaces every
per-neuron Python branch with a fixed scalar activation drawn from a small
library. Once built, the controller is a pure reservoir:

```text
X[t+1] = f( Win · I[t] + W · X[t] )
O[t+1] = Wout · g( X[t+1] )      (scalar steering command)
```

In the simulator loop, the state update uses the previous state and current
input, the readout is then computed from the updated state, and `O` is passed
to `bot.forward()`. The actuator clips that command to the allowed `[-5 deg,
+5 deg]` range.

with `n = 1000` neurons, `p = 64` color rays, `n_inputs = 2 p + 3 = 131`
(proximity[0..p-1], color[0..p-1], hit, energy, bias=1), `warmup = 0`,
`leak = 1`, and identity readout `g(x) = x`.

Verification command:

```powershell
python braincraft\env2_player_reflex_bio2.py
```

Accepted score on 2026-04-17: **14.71 +/- 0.00**.

---

## 2. Activation Library

Every hidden neuron uses a fixed scalar activation. For `z = (Win·I + W·X)[i]`,
`out[i] = f_i(z)` where `f_i` is one of:

| Name        | Formula                       | Main role                                     |
| ----------- | ----------------------------- | --------------------------------------------- |
| `relu_tanh` | `max(0, tanh(z))`             | default; thresholds, latches, AND/OR gates    |
| `identity` | `z`                           | accumulators and signed passthrough state    |
| `relu`      | `max(0, z)`                   | counters (`X19`, `X22`) that may exceed 1     |
| `clip_a`    | `clip(z, -a, a)` with `a=5 deg` | steering output `X6`                         |
| `sin`       | `sin(z)`                      | trig pair `cos_n`, `sin_n`                    |
| `square`    | `z^2`                         | exact `sin^2` magnitude test                  |
| `bump`      | `max(0, 1 - 4 z^2)`           | compact-support detectors                     |

The bump function has support `|z| < 0.5` with `bump(0) = 1`. It is used by:

- `xi_blue[r]` (centered on color value 4),
- `near_center` (centered on `X10 = 0`),
- `near_e`, `near_w` (centered on `X10 = -/+ DRIFT_OFFSET`).

Sharp logic gates use the default `relu_tanh` with a large gain
`K_SHARP = 50`, so `relu_tanh(K_SHARP · (x - thr))` saturates to ~1 for
`x > thr` and to 0 for `x < thr`.

Constants:

```text
SHORTCUT_TURN    = -2.0
SIN_HORIZ_THR    = 0.35
SIN_VERT_THR     = 0.70
COUNTER_THR      = 60
NEAR_CENTER_THR  = 0.05
DRIFT_OFFSET     = 0.175
TURN_STEPS       = 18
APPROACH_STEPS   = 45
SC_TOTAL         = TURN_STEPS + APPROACH_STEPS = 63
COLOR_EVIDENCE_THRESHOLD = 2.0
FRONT_GAIN_MAG   = 20 deg (radians)
GATE_C           = 1.0
K_SHARP          = 50
STEP_A           = 5 deg (radians)
cal_gain         = 1 / 0.173
```

---

## 3. Input Vector and Fixed Slots

The environment provides `I = [prox[0..p-1], color[0..p-1], hit, energy, 1]`.
Ray-bank aliases used below:

```text
L_idx          = 20        (left proximity tap)
R_idx          = 43        (right proximity tap, = p - 1 - L_idx)
left_side_idx  = 11        (left safety ray)
right_side_idx = 52        (right safety ray)
C1_idx, C2_idx = 31, 32    (central front proximity pair)
hit_idx        = 128
energy_idx     = 129
bias_idx       = 130
```

Neuron slots:

| Slot       | Neuron                 | Activation   |
| ---------- | ---------------------- | ------------ |
| `0..4`     | reflex features        | `relu_tanh`  |
| `5`        | unused                 | (unused)     |
| `6`        | steering `dtheta`      | `clip_a`     |
| `7`        | direction accumulator  | `identity`   |
| `8, 9`     | unused                 | (unused)     |
| `10, 11`   | `(x, y)` position      | `identity`   |
| `12`       | unused                 | (unused)     |
| `13`       | initial-heading latch  | `identity`   |
| `14`       | parity correction tap  | `identity`   |
| `15..18`   | reward circuit         | `relu_tanh`  |
| `19`       | step counter           | `relu`       |
| `20`       | `shortcut_steer`       | `identity`   |
| `22`       | shortcut countdown     | `relu`       |
| `23`       | `init_impulse`         | `identity`   |
| `24`       | seeded flag            | `relu_tanh`  |
| `25..88`   | `xi_blue[0..63]`       | `bump`       |
| `89..132`  | helpers (see §4)       | mixed        |

Helper layout (no red-fallback bank):

| Range       | Meaning                                                                             |
| ----------- | ----------------------------------------------------------------------------------- |
| `89..99`    | `l_ev, r_ev, dleft, dright, evidence, trig_pos, trig_neg, fs_pos, fs_neg, x5_pos, x5_neg` |
| `100..101`  | `cos_n, sin_n`                                                                      |
| `102..106`  | `sin_sq, cos_pos, cos_neg, y_pos, y_neg`                                            |
| `107..113`  | `near_center, heading_vert, heading_horiz, front_clear, enough_steps, sc_idle, nc_and_hv` |
| `114`       | `trig_sc`                                                                           |
| `115..117`  | `on_22, is_turn, is_app`                                                            |
| `118, 119`  | `seed_pos, seed_neg`                                                                |
| `120..123`  | `cy_pp, cy_pn, cy_np, cy_nn`                                                        |
| `124..132`  | `cos_big_pos, cos_big_neg, near_e, near_w, ncr_e, ncr_w, cos_small, ncr_c, near_corr` |

---

## 4. Main Circuits

All formulas use the convention `*_prev` to denote the value at step `t` when
computing step `t+1`. When the time index is unambiguous it is omitted.

### 4.1 Reflex feature neurons `X0..X4` (relu_tanh)

Pure feed-forward readouts of proximity/hit, with an approach-phase cancellation
term:

```text
X0 = relu_tanh( hit                       - K_SHARP·100·IS_APP_prev )
X1 = relu_tanh( prox[L_idx]               - K_SHARP·100·IS_APP_prev )
X2 = relu_tanh( prox[R_idx]               - K_SHARP·100·IS_APP_prev )
X3 = relu_tanh( -prox[left_side_idx]  + 0.75 - K_SHARP·100·IS_APP_prev )
X4 = relu_tanh( -prox[right_side_idx] + 0.75 - K_SHARP·100·IS_APP_prev )
```

The large `-K_SHARP·100·IS_APP` term silences these five source neurons while
the shortcut approach phase is active; the downstream steering output then
reduces to `front_block + shortcut_steer`.

### 4.2 Steering output `O` and `X6`

The scalar readout and its clipped lagged copy used inside the recurrence:

```text
O = Wout · g(X)
  = Wout · X
  =  hit_turn           · X0
  +  heading_gain       · X1
  + (-heading_gain)     · X2
  +  safety_gain_left   · X3
  +  safety_gain_right  · X4
  + FRONT_GAIN_MAG · X5P  - FRONT_GAIN_MAG · X5N
  + 1.0 · shortcut_steer
  + 1.0 · init_impulse

hit_turn          = -10 deg / tanh(1)
heading_gain      = -40 deg
safety_gain_left  = -20 deg
safety_gain_right = +20 deg

X6[t] = clip( O[t-1], -a, +a ),  a = 5 deg
```

`X6` is implemented by copying `Wout[0,:]` into `W[6,:]` after everything else
is wired, and applying `clip` as the activation for slot 6. This makes `X6`
the clipped delayed steering state used by the recurrent dynamics. The
simulator still forwards `O = Wout · g(X)` to `bot.forward()` each step, and
the actuator applies its own clamp there as well.

### 4.3 Direction and position accumulators

```text
X7  = X7_prev + X6_prev                                    (identity)

phi = X7 + X13 + X6
cos_n = sin( phi + pi     ) = -sin(phi)                    (sin activation)
sin_n = sin( phi + pi / 2 ) =  cos(phi)                    (sin activation)

X10 = X10_prev + speed · cos_n_prev                        (identity)
X11 = X11_prev + speed · sin_n_prev                        (identity)
```

`speed = 0.01` (from `bot.speed`). The sign of `cos_n` is negated and `sin_n`
is cosine-shaped; this preserves the downstream `X10 / X11` frame that all
corridor logic was tuned against, with only the `sin` activation available
(option 2A).

### 4.4 Initial heading correction

Let the raw correction at time `t` be:

```text
current_corr_t = (prox[R_idx] - prox[L_idx]) · cal_gain
```

Three neurons implement the latch. `X24` saturates to 1 after step 0:

```text
X24 = relu_tanh( 10 )         ~= 1  for t >= 1
```

`SEEDP` / `SEEDN` are one-step pulses that fire only while `X24_prev = 0`:

```text
SEEDP = relu_tanh( -cal_gain · prox[L_idx] + cal_gain · prox[R_idx]
                   - 1e3 · X24_prev )
SEEDN = relu_tanh(  cal_gain · prox[L_idx] - cal_gain · prox[R_idx]
                   - 1e3 · X24_prev )
```

Together `SEEDP - SEEDN` encodes the signed initial correction and closely
approximates `current_corr_0` over the observed initialization range. Two
neurons consume them:

```text
X13 = X13_prev + SEEDP_prev - SEEDN_prev                     (identity)
init_impulse = -SEEDP_prev + SEEDN_prev    (~ -current_corr_0)
```

`X13` carries the initial correction forever (a 1-step lag vs bio1, negligible).
`X14` keeps the instantaneous correction signal as a parity/legacy identity
slot, but no downstream bio2 circuit reads it. `init_impulse` is a one-step
steering pulse delivered through `Wout` with weight `1.0`.

### 4.5 Color evidence (blue bump detectors)

For each ray `r = 0..n_rays-1`:

```text
xi_blue[r] = bump( color[r] - 4 )
           = max( 0, 1 - 4 · (color[r] - 4)^2 )
```

The left/right evidence sums (half = 32):

```text
L_EV = sum_{r=0..half-1}     xi_blue[r]
R_EV = sum_{r=half..n_rays-1} xi_blue[r]
```

No `xi_red` fallback is wired in the accepted controller.

### 4.6 Gated delta pulses and signed evidence integrator

```text
DLEFT  = relu_tanh( K_SHARP · L_EV  - K_SHARP · R_EV
                    - K_SHARP · 10 · (FS_P_prev + FS_N_prev)
                    - 0.2 · K_SHARP )

DRIGHT = relu_tanh( -K_SHARP · L_EV  + K_SHARP · R_EV
                    - K_SHARP · 10 · (FS_P_prev + FS_N_prev)
                    - 0.2 · K_SHARP )

EVIDENCE = EVIDENCE_prev + DRIGHT_prev - DLEFT_prev     (identity)
```

The big `-K_SHARP·10·FS_*` term forces `DLEFT, DRIGHT -> 0` once a front sign
has latched.

### 4.7 Front-sign triggers and self-latches

```text
TP = relu_tanh(  K_SHARP · EVIDENCE_prev
                - K_SHARP · (COLOR_EVIDENCE_THRESHOLD - 0.5) )
TN = relu_tanh( -K_SHARP · EVIDENCE_prev
                - K_SHARP · (COLOR_EVIDENCE_THRESHOLD - 0.5) )

FS_P = relu_tanh( K_SHARP · FS_P_prev + K_SHARP · TP_prev )
FS_N = relu_tanh( K_SHARP · FS_N_prev + K_SHARP · TN_prev )
```

Each `FS_*` latch is self-recurrent with gain `K_SHARP`, so once set it stays
saturated at 1.

### 4.8 Front-block gating `X5_POS` / `X5_NEG`

```text
front_raw = prox[C1_idx] + prox[C2_idx]

X5P = relu_tanh( front_raw - (front_thr + GATE_C)
                 + GATE_C · FS_P_prev - GATE_C · FS_N_prev )
X5N = relu_tanh( front_raw - (front_thr + GATE_C)
                 - GATE_C · FS_P_prev + GATE_C · FS_N_prev )

front_thr = 1.4,  GATE_C = 1.0
```

Effectively `X5P` fires when the front is blocked AND `FS_P` is latched; `X5N`
mirrors for `FS_N`. These feed `Wout` with `+FRONT_GAIN_MAG` and
`-FRONT_GAIN_MAG` respectively.

### 4.9 Magnitude and sign helpers

```text
SIN_SQ   = (SIN_N_prev)^2                       (square)

COS_POS  = relu_tanh(  K_SHARP · COS_N_prev )
COS_NEG  = relu_tanh( -K_SHARP · COS_N_prev )
Y_POS    = relu_tanh(  K_SHARP · X11_prev )
Y_NEG    = relu_tanh( -K_SHARP · X11_prev )
```

### 4.10 Center and corridor bump detectors

With `bump_scale = 1 / (2 · NEAR_CENTER_THR) = 10`:

```text
NEAR_CENTER = bump( bump_scale · X10_prev )
            = bump( X10_prev / (2 · NEAR_CENTER_THR) )
            (support |X10| < NEAR_CENTER_THR = 0.05)

NEAR_E      = bump( bump_scale · X10_prev + DRIFT_OFFSET · bump_scale )
            = bump( (X10_prev + DRIFT_OFFSET) / (2 · NEAR_CENTER_THR) )

NEAR_W      = bump( bump_scale · X10_prev - DRIFT_OFFSET · bump_scale )
            = bump( (X10_prev - DRIFT_OFFSET) / (2 · NEAR_CENTER_THR) )
```

### 4.11 Heading predicates

```text
HV (heading_vert)  = relu_tanh( (K_SHARP / SIN_VERT_THR^2) · SIN_SQ_prev
                                - K_SHARP )
                   ~= 1 iff  SIN_SQ > SIN_VERT_THR^2  (0.49)

HH (heading_horiz) = relu_tanh( -(K_SHARP / SIN_HORIZ_THR^2) · SIN_SQ_prev
                                + K_SHARP )
                   ~= 1 iff  SIN_SQ < SIN_HORIZ_THR^2 (0.1225)
```

### 4.12 Shortcut support predicates

```text
FC  (front_clear)   = relu_tanh( -K_SHARP · X5P_prev - K_SHARP · X5N_prev
                                 + 0.1 · K_SHARP )
                    ~= 1 iff  X5P + X5N < 0.1

ES  (enough_steps)  = relu_tanh( (K_SHARP / COUNTER_THR) · X19_prev - K_SHARP )
                    ~= 1 iff  X19 > COUNTER_THR  (60)

SCI (sc_idle)       = relu_tanh( -K_SHARP · X22_prev + 0.5 · K_SHARP )
                    ~= 1 iff  X22 < 0.5
```

### 4.13 Step counter `X19` (relu)

```text
X19 = relu( X19_prev + 1 - 1e6 · NCV_prev )
```

A large negative pulse from `NCV` resets the counter to 0 whenever the bot is
near-center with a vertical heading.

### 4.14 `NCV`: near-center AND heading-vert (with lowered threshold)

Because `NC` is bump-shaped (tapers to 0 at the band edge), the AND uses a
bias of `-1.2·K_SHARP` instead of `-1.5·K_SHARP`:

```text
NCV = relu_tanh( K_SHARP · NC_prev + K_SHARP · HV_prev - 1.2 · K_SHARP )
```

### 4.15 `near_corr`: corridor-aware center predicate

```text
COSBP (cos_big_pos) = relu_tanh(  K_SHARP · COS_N_prev - 0.5 · K_SHARP )
                    ~= 1 iff  COS_N > 0.5

COSBN (cos_big_neg) = relu_tanh( -K_SHARP · COS_N_prev - 0.5 · K_SHARP )
                    ~= 1 iff  COS_N < -0.5

COS_SMALL = relu_tanh( -K_SHARP · COSBP_prev - K_SHARP · COSBN_prev
                       + 0.5 · K_SHARP )
          ~= 1 iff  |COS_N| <= 0.5

NCR_E = relu_tanh( K_SHARP · COSBP_prev + K_SHARP · NEAR_E_prev
                  - 1.2 · K_SHARP )
NCR_W = relu_tanh( K_SHARP · COSBN_prev + K_SHARP · NEAR_W_prev
                  - 1.2 · K_SHARP )
NCR_C = relu_tanh( K_SHARP · COS_SMALL_prev + K_SHARP · NC_prev
                  - 1.2 · K_SHARP )

NEAR_CORR = relu_tanh( K_SHARP · NCR_E_prev + K_SHARP · NCR_W_prev
                      + K_SHARP · NCR_C_prev - 0.5 · K_SHARP )
```

`NEAR_CORR` fires when the bot is near one of the three corridor centers,
choosing the drift-offset band that matches its current heading quadrant.

### 4.16 Shortcut trigger `TSC`

Six-way AND with double inhibition to prevent back-to-back firings:

```text
TSC = relu_tanh( K_SHARP · X18_prev
                 + K_SHARP · HH_prev
                 + K_SHARP · FC_prev
                 + K_SHARP · ES_prev
                 + K_SHARP · NEAR_CORR_prev
                 + K_SHARP · SCI_prev
                 - K_SHARP · 10 · TSC_prev   (blocks t+1)
                 - K_SHARP · X22_prev        (blocks t+2..t+SC_TOTAL)
                 - K_SHARP · 5.5 )
```

`TSC ~ 1` iff all six inputs are 1 AND it has not fired recently. The bias
`-5.5 K_SHARP` is the AND-of-6 threshold.

### 4.17 Countdown `X22` and phase indicators

```text
X22 = relu( X22_prev - 1 + (SC_TOTAL + 1) · TSC_prev )

ON22  = relu_tanh(  K_SHARP · X22_prev - 0.5 · K_SHARP )
IST   = relu_tanh(  K_SHARP · X22_prev - K_SHARP · (APPROACH_STEPS + 0.5) )
ISA   = relu_tanh(  K_SHARP · ON22_prev - K_SHARP · IST_prev
                   - 0.5 · K_SHARP )
```

Phase semantics: `ON22 = 1` during countdown (63 -> 0), `IST = 1` for the
first 18 steps (turn phase), `ISA = 1` for the following 45 steps (approach
phase).

### 4.18 Quadrant ANDs and `shortcut_steer`

Four 3-way ANDs with bias `-2.5·K_SHARP`:

```text
CY_PP = relu_tanh( K_SHARP · COS_POS + K_SHARP · Y_POS + K_SHARP · IST
                  - 2.5 · K_SHARP )
CY_PN = relu_tanh( K_SHARP · COS_POS + K_SHARP · Y_NEG + K_SHARP · IST
                  - 2.5 · K_SHARP )
CY_NP = relu_tanh( K_SHARP · COS_NEG + K_SHARP · Y_POS + K_SHARP · IST
                  - 2.5 · K_SHARP )
CY_NN = relu_tanh( K_SHARP · COS_NEG + K_SHARP · Y_NEG + K_SHARP · IST
                  - 2.5 · K_SHARP )
```

(All inputs to each `CY_*` are read as `*_prev`.) The steering channel:

```text
shortcut_steer = |SHORTCUT_TURN| · ( CY_PN_prev + CY_NP_prev
                                   - CY_PP_prev - CY_NN_prev )
```

This encodes the turn rule `turn_toward = -sign(cos) · sign(y)` with magnitude
`|SHORTCUT_TURN| = 2.0`, and is gated to the turn phase by `IST`.

### 4.19 Reward circuit `X15..X18`

With constants `K = 0.005`, `arm_from_energy = 1000`, `arm_latch = 10`,
`pulse_gain = 1e5`, `pulse_thr = 0.2`, `arm_gate = 1000`, `latch_gain = 10`:

```text
X15 = relu_tanh( K · energy )

X16 = relu_tanh( arm_from_energy · X15_prev + arm_latch · X16_prev )

X17 = relu_tanh( pulse_gain · K · energy
                - pulse_gain · X15_prev
                + arm_gate · X16_prev
                - (arm_gate + pulse_thr) )

X18 = relu_tanh( latch_gain · X17_prev + latch_gain · X18_prev )
```

`X18` is the armed reward latch consumed by the shortcut trigger `TSC`.

---

## 5. Output Weight Table (`Wout`)

Only these slots are nonzero:

```text
Wout[0, 0]               =  hit_turn            = -10 deg / tanh(1)
Wout[0, 1]               =  heading_gain        = -40 deg
Wout[0, 2]               = -heading_gain        = +40 deg
Wout[0, 3]               =  safety_gain_left    = -20 deg
Wout[0, 4]               =  safety_gain_right   = +20 deg
Wout[0, X5P]             = +FRONT_GAIN_MAG      = +20 deg
Wout[0, X5N]             = -FRONT_GAIN_MAG      = -20 deg
Wout[0, shortcut_steer]  = +1.0
Wout[0, init_impulse]    = +1.0
```

The clipped steering neuron `X6` reads the same row of `Wout` through a copy
into `W[6, :]`, so `z[6] = O_{t-1}` and the recurrent state stores
`X6 = clip(O_{t-1}, -a, a)`. The simulator then forwards the fresh readout
`O = Wout · g(X)` to the bot each step, with the actuator clip applied in
`bot.forward()`.

---

## 6. Verification

The accepted controller passes all three smoke tests:

```powershell
python braincraft\_debug_bio2.py --steps 120 --stride 20
python braincraft\_debug_bio2_detail.py --steps 120
python braincraft\env2_player_reflex_bio2.py
```

with a final env2 score of `14.71 +/- 0.00`.
