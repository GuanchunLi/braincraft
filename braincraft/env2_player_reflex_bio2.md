# Reflex Bio Player 2 — Environment 2

## Overview

`Reflex Bio Player 2` (`env2_player_reflex_bio2.py`) is a **biologically
plausible** variant of `env2_player_reflex_bio.py`. Both controllers solve
the same task — clockwise wall-following around a ring arena with a
corridor shortcut after the first reward — but bio2 obeys a much stricter
constraint:

> **The activation function `f` is a purely pointwise per-neuron map.**
> Each `out[i]` depends only on `x[i]` and a fixed per-neuron choice
> drawn from a small library of scalar functions (`relu_tanh`, `identity`,
> `relu`, `clip`, `sin`, `cos`).  No multi-neuron logic — no gating, no
> latches, no `if`-statements — is allowed inside `f`.

All circuits that bio1 implemented as Python conditionals inside its
custom activation (gating, latches, state machines, trigonometry, position
updates, shortcut switching, initial-heading depletion) are reformulated
in bio2 as **networks of helper neurons** wired through `Win`, `W`, and
`Wout`.  The network behaves as a pure reservoir:

```
X(t) = f( W_in · I(t)  +  W · X(t−1) )
```

Logical AND / OR / NOT / latches / thresholds are implemented as
sharp-gain `relu_tanh` neurons that approximate Heaviside functions.

Performance: bio2 reaches ≈ **14.71** mean distance vs bio1's **14.83**
under the same evaluation harness — a small gap that comes mainly from a
1–2 step extra delay in the multi-stage trigger pipelines.

---

## 1. Network Architecture

| Component | Symbol | Dimensions |
|-----------|--------|------------|
| Input weights      | `W_in`  | 1000 × 131 |
| Recurrent weights  | `W`     | 1000 × 1000 |
| Output weights     | `W_out` | 1 × 1000 |
| Hidden state       | `X`     | 1000 × 1 |
| Input vector       | `I`     | 131 × 1 |
| Output             | `O`     | scalar (steering Δθ) |

### 1.1 Input Vector

Camera resolution `p = 64`, so `I` has `2p + 3 = 131` dimensions:

```
I = [ 1 − d_0, ..., 1 − d_63,    c_0, ..., c_63,    hit, energy, 1 ]ᵀ
       depths (64)                colors (64)        scalars (3)
```

- `d_i ∈ [0, 1]` — depth of camera ray `i` (inverted so closer = higher).
- `c_i` — colour-index value of camera ray `i` (blue is high).
- `hit` — collision flag from previous step.
- `energy` — current bot energy.
- `1.0` — constant bias.

Useful index aliases (used below):

```
hit_idx    = 128         L_idx          = 20      C1_idx = 31
energy_idx = 129         R_idx          = 43      C2_idx = 32
bias_idx   = 130         left_side_idx  = 11
                          right_side_idx = 52
```

### 1.2 State Update

Leak `λ = 1`, so:

```
X(t) = f( W_in · I(t) + W · X(t−1) )
O(t) = W_out · X(t)
```

The output `O(t)` is fed to `bot.forward(O, env)`.  Because `X[6]` is
the steering output stored in the state, **every output is delayed by one
step relative to its source neurons**.  This 1-step lag is consistent
with bio1.

---

## 2. Pointwise Activation Function `f`

Every neuron uses one of six scalar functions, looked up per-index.
Let `z = W_in · I + W · X(t−1)`.

| Function   | Formula                          | Used by                                      |
|------------|----------------------------------|----------------------------------------------|
| `relu_tanh`| `max(0, tanh(z))`                | default — all sensors, latches, gates        |
| `identity` | `z`                              | accumulators: `X7, X10, X11, X13, X14, X20, X23, X_evidence` |
| `relu`     | `max(0, z)`                      | unbounded counters: `X19, X22`               |
| `clip`     | `clip(z, −a, a)` with `a = 5°`   | steering output `X6`                         |
| `cos`      | `cos(z)`                         | `X_cos`                                      |
| `sin`      | `sin(z)`                         | `X_sin`                                      |

That is the **entire** list — every conditional in the bio1 activation
is gone, replaced by extra rows of `Win` / `W`.

---

## 3. Sharp-Threshold Logic

Two constants underpin the boolean circuits:

```
K_SHARP = 50           # gain so tanh saturates near |z| > 0.1
STEP_A  = 5° (radians) # steering clip for X6
```

For a `relu_tanh` neuron with input `z = K_SHARP · (s − τ)`:

```
output ≈ 1   when s > τ + ε
output ≈ 0   when s < τ − ε
```

Composition tricks (used throughout):

```
AND(a, b)    →   relu_tanh( K · (a + b − 1.5) )
AND(a, b, c) →   relu_tanh( K · (a + b + c − 2.5) )
OR(a, b)     →   relu_tanh( K · (a + b − 0.5) )
NOT(a)       →   relu_tanh( K · (0.5 − a) )
THR(s, τ)    →   relu_tanh( K · (s − τ) )      (output ≈ 1 iff s > τ)
LATCH        →   relu_tanh( K · (X_self + trigger) )  (self-recurrence)
```

These are the building blocks of every gate below.

---

## 4. Neuron Index Layout

Bio2 uses a sequential allocator.  With `p = n_rays = 64`:

```
X0..X4    : reflex sensor features (relu_tanh)
X5        : front-block, kept for diagnostics only (no Wout entry)
X6        : steering Δθ output           (clip ±a)
X7        : direction accumulator        (identity)
X8        : (unused — bio1 cos slot)
X9        : (unused — bio1 sin slot)
X10, X11  : position x, y                (identity)
X12       : hit copy                     (relu_tanh)
X13       : locked initial-heading correction (identity)
X14       : instantaneous correction  (identity, kept for parity)
X15..X18  : reward circuit               (relu_tanh)
X19       : step counter                 (relu)
X20       : shortcut steering channel    (identity)
X22       : shortcut countdown           (relu)
X23       : initial-correction remainder (identity)
X24       : seeded flag                  (relu_tanh)
X25  .. X88   : Xi_hi  (per-ray colour-high detector)
X89  .. X152  : Xi_lo  (per-ray colour-low  detector)
X153 ..       : bio2 helpers (allocated by `_bio2_indices`):

   l_ev, r_ev,  dleft, dright,  evidence,
   trig_pos, trig_neg,  fs_pos, fs_neg,  x5_pos, x5_neg,
   cos_n, sin_n,
   sin_pos, sin_neg, cos_pos, cos_neg, y_pos, y_neg, x_pos, x_neg,
   near_center, heading_vert, heading_horiz, front_clear,
   enough_steps, sc_idle, nc_and_hv, trig_sc,
   on_22, is_turn, is_app,
   tt_plus, tt_minus,
   x23p, x23n, is_init,
   seed_pos, seed_neg,
   cy_pp, cy_pn, cy_np, cy_nn,
   cos_big_pos, cos_big_neg,
   xe_pos, xe_neg, xw_pos, xw_neg,
   ncr_e, ncr_w, cos_small, ncr_c, near_corr
```

The total live count is well under 1000.

---

## 5. Reflex Feature Neurons (`X0` – `X5`)

These reproduce the wall-following primitives of `reflex3`.

```
X0 = relu_tanh( hit )                                — collision flag
X1 = relu_tanh( I[L_idx]  )           ≈  prox-left  ray (index 20)
X2 = relu_tanh( I[R_idx]  )           ≈  prox-right ray (index 43)
X3 = relu_tanh( −I[11] + 0.75 )       — left  safety, target offset 0.75
X4 = relu_tanh( −I[52] + 0.75 )       — right safety, target offset 0.75
X5 = relu_tanh( I[31] + I[32] − 1.4 ) — front-block (kept for telemetry)
```

The corresponding `W_out` row (heading + safety steering):

```
W_out[0, 0] = −10° / tanh(1)   (hit_turn)
W_out[0, 1] = −40°             (heading_gain · X1)
W_out[0, 2] = +40°             (heading_gain · X2 with sign flip)
W_out[0, 3] = −20°             (safety_gain_left  · X3)
W_out[0, 4] = +20°             (safety_gain_right · X4)
W_out[0, 5] =   0              (front_block disabled — replaced by X5p/X5n)
```

### 5.1 Approach-phase cancellation

Bio1 does `x20 = −O_no_front` while in the shortcut **approach** phase to
silence reflexes.  Bio2 achieves the same by **directly suppressing X0–X4**
when the approach indicator `IS_APP` is high:

```
W[i, IS_APP] = −K_SHARP · 100          for i ∈ {0, 1, 2, 3, 4}
```

The huge negative weight forces `relu_tanh` to 0 whenever `IS_APP ≈ 1`,
so the only contribution to `O` during the approach is the front-block
channel and `X20` itself — matching bio1's behaviour.

---

## 6. Steering Output `X6`

`X6` is the clipped sum of all `W_out` contributions, computed with a
1-step lag.  Bio2 builds this by **copying `W_out` row 0 into `W` row 6**
after all output weights have been wired:

```
W[6, j] = W_out[0, j]    for every j with W_out[0, j] ≠ 0
```

Then with the `clip(z, −a, a)` activation:

```
X6(t) = clip( Σ_j W_out[0, j] · X_j(t−1),  −a,  +a ),   a = 5°
```

`X6` is fed back to the trig neurons and to the direction accumulator
(see §7).

---

## 7. Heading, Trig, and Position

### 7.1 Direction accumulator

```
X7(t) = X7(t−1) + X6(t−1)
```

(`identity` activation; `W[7, 7] = W[7, 6] = 1`.)

### 7.2 Trigonometric neurons

Both `X_cos` and `X_sin` share the same pre-activation:

```
z_cos = z_sin = X7 + X13 + X6 + π/2
X_cos = cos( z_cos )
X_sin = sin( z_sin )
```

This effectively computes `cos(θ)` and `sin(θ)` where:

```
θ_now = X7 + X13 + X6 + π/2
      = (Σ Δθ)  +  init_correction  +  current_Δθ  +  π/2
```

The added `+ X6` brings the current heading update *into* the trig
arguments, so position is integrated using the **post-update** heading
— matching bio1's "current θ" usage.

### 7.3 Position integration

```
X10(t) = X10(t−1) + speed · X_cos(t−1)        speed = 0.01
X11(t) = X11(t−1) + speed · X_sin(t−1)
```

(`identity` activations; no hit-gating because transient drift during
collisions is negligible at `speed = 0.01`.)

### 7.4 Hit copy

`X12(t) = relu_tanh( hit )` — diagnostic only.

---

## 8. Initial Heading Correction

The bot starts with direction `π/2 + ε`, `ε ~ U(−5°, +5°)`.  Bio2 must
realign without invoking any Python `if`-test on step number.

### 8.1 The `seed_pos` / `seed_neg` one-shots

These two `relu_tanh` neurons capture the initial sensor asymmetry, but
**only on step 1**, by gating themselves on the seeded flag `X24`:

```
cal_gain = 1 / 0.173

z(seed_pos) = −cal_gain · I[L_idx] + cal_gain · I[R_idx] − 1000 · X24(t−1)
z(seed_neg) = +cal_gain · I[L_idx] − cal_gain · I[R_idx] − 1000 · X24(t−1)

seed_pos  = relu_tanh( z(seed_pos) )
seed_neg  = relu_tanh( z(seed_neg) )
```

At step 0, `X24(t−1) = 0`, so `seed_pos ≈ tanh( current_corr )` and
`seed_neg ≈ tanh( −current_corr )`.  The huge negative weight on `X24`
suppresses both forever once the seed flag latches.

### 8.2 The seeded flag `X24`

```
z(X24) = 10
X24    = relu_tanh( 10 ) ≈ 1   — saturates from step 0 onward
```

The 1-step state delay means `X24(t−1) = 0` at step 0 and `≈ 1` from
step 1 on, gating `seed_pos / seed_neg` correctly.

### 8.3 The `X13` latch (locked initial correction)

```
X13(t) = X13(t−1) + seed_pos(t−1) − seed_neg(t−1)
```

Self-recurrence preserves the value once the seeds stop firing, so
`X13` carries the captured `current_corr_0 ≈ ±tanh(small)` for the rest
of the run — exactly the role bio1's `X13` plays.  (One step of lag
versus bio1; impact negligible.)

### 8.4 The `X23` remainder + depletion

`X23` is the remaining initial Δθ.  It is:

- **Seeded** on step 0 by `seed_pos` / `seed_neg` (negative sign so the
  sign matches bio1's `clip(−correction, ±CAP)`).
- **Depleted** by `±a` per step via the sign branches `x23p` / `x23n`.

```
W[23, 23]   = +1
W[23, seed_pos] = −1
W[23, seed_neg] = +1
W[23, x23p]    = −a       a = 5°
W[23, x23n]    = +a

X23(t) = X23(t−1) − seed_pos(t−1) + seed_neg(t−1)
                  − a · x23p(t−1) + a · x23n(t−1)
```

Sign branches (sharp threshold around `±ε = 1e-3`):

```
x23p = relu_tanh(  K_INIT · X23 ),    K_INIT = 1 / INIT_CORR_EPS
x23n = relu_tanh( −K_INIT · X23 )
```

So `x23p ≈ 1` while `X23 > ε`, depleting `X23` by `a`/step until it
falls below `ε`; symmetric for the negative side.

### 8.5 One-shot init steering pulse

Bio1's depletion contributes to steering on every step.  Bio2 instead
delivers the entire `−current_corr_0` impulse to the steering channel
**once**, at step 1, via the same seeds:

```
W[20, seed_pos] = −1
W[20, seed_neg] = +1
```

Because `seed_pos / seed_neg` only fire on step 0 (before `X24`
latches), the impulse appears in `X20` exactly once — eliminating the
need to read a stateful integrator on every step.

### 8.6 Optional `is_init` indicator

```
is_init = relu_tanh( K · ( x23p + x23n + X24 − 1.5 ) )
```

= "X23 still has magnitude AND we are seeded" — kept for diagnostics.

---

## 9. Colour Evidence Pipeline

The bot must learn from the blue wall on which side the corridor lies.
Bio2 computes the same evidence integrator as bio1 but using
sharp-gate `relu_tanh` neurons.

### 9.1 Per-ray detectors `Xi_hi`, `Xi_lo`

For every ray `r ∈ [0, 64)`, with thresholds `BLUE_HI_THR = 3.5`,
`BLUE_LO_THR = 4.5`:

```
Xi_hi[r] = relu_tanh( c_r − 3.5 )      ≈ 1 if  c_r > 3.5
Xi_lo[r] = relu_tanh( c_r − 4.5 )      ≈ 1 if  c_r > 4.5
```

(Layered so `Xi_hi − XI_LO_FACTOR · Xi_lo` peaks for blue tones, see
below.)

### 9.2 Side aggregates `L_EV` / `R_EV`

Let `half = 32`.  The detectors on each half are summed with a low
weight on `Xi_lo` to suppress over-saturated rays:

```
L_EV = relu_tanh( Σ_{r=0..31}   ( Xi_hi[r] − 3 · Xi_lo[r] ) )
R_EV = relu_tanh( Σ_{r=32..63}  ( Xi_hi[r] − 3 · Xi_lo[r] ) )
```

With `XI_LO_FACTOR = 3.0`.

### 9.3 Gated delta pulses `dleft` / `dright`

These become 1 only when one side dominates AND the front-sign latch
`fs_pos / fs_neg` is still 0:

```
z(dleft)  = K_SHARP · (  L_EV − R_EV )
            − 10 · K_SHARP · ( fs_pos + fs_neg )
            − 0.2 · K_SHARP

z(dright) = K_SHARP · ( −L_EV + R_EV )
            − 10 · K_SHARP · ( fs_pos + fs_neg )
            − 0.2 · K_SHARP

dleft  = relu_tanh( z(dleft) )
dright = relu_tanh( z(dright) )
```

Once a sign is latched, both pulses are suppressed forever.

### 9.4 Signed integrator `evidence`

```
evidence(t) = evidence(t−1) + dright(t−1) − dleft(t−1)
```

(`identity` activation.)

### 9.5 Trigger thresholds `trig_pos` / `trig_neg`

Threshold is offset by 0.5 so that `evidence == COLOR_EVIDENCE_THRESHOLD`
itself fires (bio1 used `>=`):

```
T = COLOR_EVIDENCE_THRESHOLD − 0.5,    COLOR_EVIDENCE_THRESHOLD = 2.0

trig_pos = relu_tanh( K_SHARP · ( evidence − T ) )
trig_neg = relu_tanh( K_SHARP · ( −evidence − T ) )
```

### 9.6 Self-latching front-sign flags `fs_pos` / `fs_neg`

```
fs_pos = relu_tanh( K_SHARP · ( fs_pos(t−1) + trig_pos(t−1) ) )
fs_neg = relu_tanh( K_SHARP · ( fs_neg(t−1) + trig_neg(t−1) ) )
```

Once either latches to ≈ 1, it self-sustains.

---

## 10. Front-Block Steering (`x5_pos` / `x5_neg`)

These two neurons split the front-block into two signed channels gated
by the latched sign.  With `front_thr = 1.4`, `GATE_C = 1`:

```
z(x5_pos) = c_31 + c_32 − (front_thr + GATE_C)
            + GATE_C · fs_pos − GATE_C · fs_neg

z(x5_neg) = c_31 + c_32 − (front_thr + GATE_C)
            − GATE_C · fs_pos + GATE_C · fs_neg

x5_pos = relu_tanh( z(x5_pos) )       — fires only if fs_pos AND front-block
x5_neg = relu_tanh( z(x5_neg) )       — fires only if fs_neg AND front-block
```

Output contribution (`FRONT_GAIN_MAG = 20°`):

```
W_out[0, x5_pos] = +20°
W_out[0, x5_neg] = −20°
```

Net effect: when blue is on the right (`fs_pos = 1`) and a frontal wall
appears, steer +20°; when blue is on the left, steer −20°.  Before any
sign is latched, both `x5_*` stay at 0.

---

## 11. Reward Detection (`X15` – `X18`)

Identical wiring to bio1.  With `K = 0.005`:

```
X15(t) = relu_tanh( K · I[energy] )

X16(t) = relu_tanh(  arm_from_energy · X15(t−1)
                    + arm_latch        · X16(t−1) )
        with arm_from_energy = 1000, arm_latch = 10

X17(t) = relu_tanh(  pulse_gain · K · I[energy]
                    − pulse_gain        · X15(t−1)
                    + arm_gate          · X16(t−1)
                    − ( arm_gate + pulse_thr ) )
        with pulse_gain = 1e5, arm_gate = 1000, pulse_thr = 0.2

X18(t) = relu_tanh( latch_gain · X17(t−1) + latch_gain · X18(t−1) )
        with latch_gain = 10
```

`X18` is a self-latching "reward has been seen" flag — fires (≈ 1) the
first time the energy increases and stays at 1 thereafter.

---

## 12. Step Counter (`X19`)

`X19` counts steps since the bot last crossed the corridor centre.
Activation is `relu`, recurrence is +1 per step with a huge negative
reset pulse driven by `nc_and_hv` (near-centre AND heading-vertical):

```
z(X19) = X19(t−1) + 1 − 1e6 · nc_and_hv(t−1)
X19    = max( 0, z(X19) )
```

So `X19 = 0` whenever the bot is centred and heading vertically;
otherwise it grows by 1.

---

## 13. Geometric Predicates

These are the inputs to the shortcut trigger.  All use sharp `relu_tanh`
gates.

### 13.1 |sin|, |cos|, |x|, |y| via paired one-sided rectifiers

For low-gain (smooth across `[0, 1]`):

```
sin_pos = relu_tanh(  1.0 · sin )      sin_neg = relu_tanh( −1.0 · sin )
|sin| ≈ sin_pos + sin_neg
```

For sharp sign signals:

```
cos_pos = relu_tanh(  K_SHARP · cos )
cos_neg = relu_tanh( −K_SHARP · cos )

y_pos   = relu_tanh(  K_SHARP · X11 )
y_neg   = relu_tanh( −K_SHARP · X11 )
```

For the "near-centre" predicate (`NEAR_CENTER_THR = 0.05`):

```
K_POS  = K_SHARP / NEAR_CENTER_THR

x_pos = relu_tanh( K_POS · X10 − K_SHARP )   ≈ 1 iff X10 >  +0.05
x_neg = relu_tanh( −K_POS · X10 − K_SHARP )  ≈ 1 iff X10 <  −0.05
```

### 13.2 Boolean predicates

```
near_center   (NC) = relu_tanh( K · ( 0.5 − x_pos − x_neg ) )
                   ≈ NOT(x_pos) AND NOT(x_neg)

heading_vert  (HV) = relu_tanh(   K/τ_v · (sin_pos + sin_neg) − K )
                   ≈ 1  iff  |sin| > 0.70
   τ_v = tanh( SIN_VERT_THR )   ≈ 0.604

heading_horiz (HH) = relu_tanh( −K/τ_h · (sin_pos + sin_neg) + K )
                   ≈ 1  iff  |sin| < 0.35
   τ_h = tanh( SIN_HORIZ_THR )  ≈ 0.336

front_clear   (FC) = relu_tanh( 0.1 · K − K · (x5_pos + x5_neg) )
                   ≈ 1  iff  x5_pos + x5_neg < 0.1

enough_steps  (ES) = relu_tanh( K/COUNTER_THR · X19 − K )
                   ≈ 1  iff  X19 > COUNTER_THR = 60

sc_idle       (SCI) = relu_tanh( 0.5 · K − K · X22 )
                    ≈ 1  iff  X22 < 0.5

nc_and_hv     (NCV) = AND(NC, HV)
                    = relu_tanh( K · ( NC + HV − 1.5 ) )
```

### 13.3 Drift-corrected near-corridor predicate `near_corr`

This is the most subtle gate.  Bio1 uses `|X10 − δ| < 0.05` with
`δ = −0.115 · sign(cos θ)`.  Bio2 must lead bio1 by ~6 reservoir steps
because the trigger pipeline (XE → NCR → NEAR_CORR → TSC → X22 → IST →
CY → X20) spans ~6 steps; therefore `DRIFT_OFFSET = 0.175` (not 0.115).

```
DRIFT_OFFSET    = 0.175
NEAR_CENTER_THR = 0.05

cos_big_pos = relu_tanh(  K · cos − 0.5 · K )   ≈ 1 iff  cos >  +0.5
cos_big_neg = relu_tanh( −K · cos − 0.5 · K )   ≈ 1 iff  cos <  −0.5
cos_small   = relu_tanh( K · ( 0.5 − cos_big_pos − cos_big_neg ) )
            = NOT(cos_big_pos) AND NOT(cos_big_neg)

xe_pos = relu_tanh( K_POS · X10 + K_POS · ( DRIFT − THR ) )
       ≈ 1  iff  X10 + DRIFT >  +THR

xe_neg = relu_tanh( −K_POS · X10 − K_POS · ( DRIFT + THR ) )
       ≈ 1  iff  X10 + DRIFT <  −THR

xw_pos = relu_tanh(  K_POS · X10 − K_POS · ( DRIFT + THR ) )
       ≈ 1  iff  X10 − DRIFT >  +THR

xw_neg = relu_tanh( −K_POS · X10 + K_POS · ( DRIFT − THR ) )
       ≈ 1  iff  X10 − DRIFT <  −THR

ncr_e = relu_tanh( K · ( cos_big_pos − xe_pos − xe_neg − 0.5 ) )
      ≈  cos_big_pos AND NOT(xe_pos) AND NOT(xe_neg)
      ≈  cos > +0.5  AND  |X10 + DRIFT| < THR     (east band)

ncr_w = relu_tanh( K · ( cos_big_neg − xw_pos − xw_neg − 0.5 ) )
      ≈  cos < −0.5  AND  |X10 − DRIFT| < THR     (west band)

ncr_c = AND( cos_small, NC )
      = relu_tanh( K · ( cos_small + NC − 1.5 ) )      (vertical fallback)

near_corr = OR( ncr_e, ncr_w, ncr_c )
          = relu_tanh( K · ( ncr_e + ncr_w + ncr_c − 0.5 ) )
```

---

## 14. Shortcut Trigger `trig_sc`

`trig_sc` is the AND of six predicates plus a refractory inhibitor:

```
z(trig_sc) = K · ( X18 + HH + FC + ES + near_corr + sc_idle  −  5.5 )
             − 10 · K · trig_sc(t−1)         (1-step refractory)
             −      K · X22(t−1)             (countdown refractory)

trig_sc = relu_tanh( z(trig_sc) )
```

Two independent inhibition paths are needed because `sc_idle` lags
`trig_sc` by 2 reservoir steps (TSC → X22 → SCI), so a single path would
allow `trig_sc` to fire twice in a row.

The 6-input AND fires (≈ 1) only when:

```
X18 ≈ 1            (reward seen)
HH  ≈ 1            (heading horizontal,  |sin| < 0.35)
FC  ≈ 1            (no front block,  x5_pos + x5_neg < 0.1)
ES  ≈ 1            (X19 > 60)
near_corr ≈ 1      (drift-corrected near corridor)
sc_idle  ≈ 1       (X22 < 0.5)
```

---

## 15. Shortcut Countdown `X22`

`X22` is a `relu` neuron implementing a one-shot decrementing counter:

```
SC_TOTAL = TURN_STEPS + APPROACH_STEPS = 18 + 45 = 63

z(X22) = X22(t−1) − 1 + (SC_TOTAL + 1) · trig_sc(t−1)
X22    = max( 0, z(X22) )
```

So when `trig_sc` fires once, `X22` jumps to `SC_TOTAL = 63` and then
decrements by 1 per step.

### 15.1 Phase indicators

```
on_22   = relu_tanh( K · X22 − 0.5 K )                ≈ 1  iff X22 > 0.5
is_turn = relu_tanh( K · X22 − K · (APPROACH_STEPS + 0.5) )
                                                       ≈ 1  iff X22 > 45.5
is_app  = relu_tanh( K · ( on_22 − is_turn − 0.5 ) )
        = AND( on_22, NOT(is_turn) )                   ≈ 1  iff 0.5 < X22 ≤ 45.5
```

So during `X22 ∈ (45.5, 63]` the bot is **turning**, and during
`X22 ∈ (0.5, 45.5]` it is **approaching**.

---

## 16. Turn-Direction Quadrants (`cy_*`, `tt_*`)

The shortcut turn direction follows `−sign(cos θ) · sign(y)`.  Bio2
decomposes this into four AND-of-three gates:

```
cy_pp = AND( cos_pos, y_pos, is_turn )    →  cos > 0, y > 0  →  TTM (−1)
cy_pn = AND( cos_pos, y_neg, is_turn )    →  cos > 0, y < 0  →  TTP (+1)
cy_np = AND( cos_neg, y_pos, is_turn )    →  cos < 0, y > 0  →  TTP (+1)
cy_nn = AND( cos_neg, y_neg, is_turn )    →  cos < 0, y < 0  →  TTM (−1)
```

Each is implemented as

```
cy_xy = relu_tanh( K · ( a + b + c − 2.5 ) )
```

where `(a, b, c)` are the three input flags.

The `tt_plus` and `tt_minus` neurons are kept for diagnostics:

```
tt_plus  = OR( cy_pn, cy_np ) = relu_tanh( K · ( cy_pn + cy_np − 0.5 ) )
tt_minus = OR( cy_pp, cy_nn ) = relu_tanh( K · ( cy_pp + cy_nn − 0.5 ) )
```

But the actual steering shortcut is wired **directly** from the four
quadrant gates into `X20` to remove one stage of lag (see §17).

---

## 17. Shortcut Steering Channel `X20`

`X20` (identity activation) carries the shortcut override and the
one-shot init impulse:

```
X20(t) =  +|SHORTCUT_TURN| · cy_pn(t−1)
        +|SHORTCUT_TURN| · cy_np(t−1)
        −|SHORTCUT_TURN| · cy_pp(t−1)
        −|SHORTCUT_TURN| · cy_nn(t−1)
        −1 · seed_pos(t−1)
        +1 · seed_neg(t−1)
```

with `SHORTCUT_TURN = −2.0` (so `|SHORTCUT_TURN| = 2.0`).

`W_out[0, X20] = 1`, so `X20` is added directly to the steering output.

Because the `seed_*` pulses fire only on step 0, the second pair of
terms delivers `−current_corr_0` to steering on step 1 — a one-shot
initial heading impulse equivalent to bio1's `clip(−correction, ±CAP)`
on the first effective step.

---

## 18. Final Output

The final steering value is the combined contribution of all `W_out`
entries, with the 1-step lag built into the `X6` clip:

```
O(t) = Σ_j W_out[0, j] · X_j(t)

     = hit_turn        · X0
       + heading_gain  · ( X2 − X1 )                 (from W[0,1]=−40°, W[0,2]=+40°)
       + safety_gain   · ( X4 − X3 )                 (signs flipped)
       + FRONT_GAIN    · ( x5_pos − x5_neg )
       + 1.0           · X20

X6(t+1) = clip( O(t),  −5°,  +5° )
```

`bot.forward(O, env)` then applies the steering and advances by
`speed = 0.01`.

---

## 19. Configurable Constants

Collected for reference (all radians for angles unless noted):

| Constant | Value | Role |
|----------|-------|------|
| `STEP_A`            | `5°`            | Δθ clip on `X6` |
| `K_SHARP`           | `50`            | Sharp-gate gain (logic) |
| `BLUE_HI_THR`       | `3.5`           | `Xi_hi` threshold |
| `BLUE_LO_THR`       | `4.5`           | `Xi_lo` threshold |
| `XI_LO_FACTOR`      | `3.0`           | Suppress over-saturated rays |
| `COLOR_EVIDENCE_THRESHOLD` | `2.0`    | `evidence` trigger level |
| `GATE_C`            | `1.0`           | `x5_*` gating amplitude |
| `FRONT_GAIN_MAG`    | `20°`           | Front-block steering output |
| `SHORTCUT_TURN`     | `−2.0`          | Magnitude for `cy_*` → `X20` |
| `SIN_HORIZ_THR`     | `0.35`          | `HH` threshold |
| `SIN_VERT_THR`      | `0.70`          | `HV` threshold |
| `NEAR_CENTER_THR`   | `0.05`          | `X10` band half-width |
| `DRIFT_OFFSET`      | `0.175`         | Bio2-specific lookahead (vs bio1's 0.115) |
| `COUNTER_THR`       | `60`            | `ES` step threshold |
| `TURN_STEPS`        | `18`            | Shortcut turn duration |
| `APPROACH_STEPS`    | `45`            | Shortcut approach duration |
| `SC_TOTAL`          | `63`            | `X22` initial value |
| `INIT_CORR_EPS`     | `1e−3`          | `X23` depletion deadband |
| `INIT_CORR_CAP`     | `0.15`          | (parameter parity with bio1) |
| `cal_gain`          | `1 / 0.173`     | Sensor → heading scale |
| `speed`             | `0.01`          | Bot speed |
| `n`                 | `1000`          | Reservoir size |
| `p = n_rays`        | `64`            | Camera resolution |
| `warmup`            | `0`             | No warmup |
| `leak`              | `1.0`           | Pure substitution |

---

## 20. Bio1 vs Bio2 — Compensations Required

The strict pointwise-`f` constraint introduces extra reservoir delay in
multi-stage circuits.  Two non-trivial compensations were needed:

1. **`DRIFT_OFFSET = 0.175`** instead of bio1's `0.115`.
   The shortcut-trigger chain
   `XE → NCR → NEAR_CORR → TSC → X22 → IST → cy_* → X20`
   spans ~6 reservoir steps.  Spatial bias must lead by
   `~6 · speed = 0.06`, giving `0.115 + 0.06 ≈ 0.175`.
   Empirically `DRIFT_OFFSET ∈ [0.16, 0.225]` all give the optimum;
   `0.115` collapses to ~3.05; `0.25` collapses to ~4.91.

2. **`W[0..4, IS_APP] = −K_SHARP · 100`** — approach-phase reflex
   silencing.  Bio1 sets `x20 = −O_no_front` during approach to cancel
   reflex contributions.  Bio2 instead suppresses the source neurons
   themselves while `IS_APP ≈ 1`.  Without this, the bot wiggles after
   the shortcut turn and hits the east wall around `t ≈ 180`.

These two compensations close most of the bio1 ↔ bio2 score gap.

---

## 21. Pipeline Summary

The end-to-end signal flow per step:

```
                      ┌───────── camera I (depths, colors, hit, energy, 1) ─────────┐
                      │                                                              │
sensors  →  X0..X4    │                                                              │
                      │                                                              │
colour rays  →  Xi_hi, Xi_lo                                                         │
                       └─→  L_EV, R_EV  →  dleft, dright  →  evidence                │
                                                                  └─→  trig_pos/neg  │
                                                                            └─→ fs_pos/neg
                                                                                     │
                                                                                     │
front rays + fs  →  x5_pos, x5_neg                                                   │
                                                                                     │
energy           →  X15..X18  (reward latch)                                         │
                                                                                     │
heading state    →  X7  →  cos_n, sin_n  →  position X10, X11                        │
                                          └→  trig predicates (NC, HV, HH, FC, ES)   │
                                                                                     │
geometric        →  near_corr (DRIFT-shifted)                                        │
                                                                                     │
trigger          →  trig_sc  →  X22  →  is_turn / is_app                             │
                                                                                     │
turn direction   →  cy_*  →  X20                                                     │
                                                                                     │
steering output  →  Σ W_out · X  →  O →  X6 (clip ±5°) →  bot.forward(O, env)        │
                                                                                     │
                                                       (state delayed by 1 step)     │
```

The whole network is purely pointwise at the activation level; every
"computation" lives in a weight matrix entry.
