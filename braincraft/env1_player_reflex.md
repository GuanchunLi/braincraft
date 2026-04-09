# Reflex Player — Technical Description

Deterministic Task 1 player that combines a clockwise wall-following
steering controller (from dummy2/dummy4) with two independent internal
estimation subsystems: **head-direction & position tracking** (from
dummy4) and **reward-state detection** (from dummy3).  In this initial
version the output depends only on the six feature neurons; all internal
state neurons have zero output weight (`Wout[0, 6:] == 0`).


## Framework

The player is a reservoir-computing model with 1000 neurons.  At each
time step the dynamics are:

```
X(t+1) = f( Win @ I(t) + W @ X(t) )        (leak = 1)
O(t)   = Wout @ X(t)                        (g = identity)
```

where `I(t)` is a 67-element input vector (64 camera rays + hit + energy
+ bias=1) and `O(t)` is a scalar steering command in radians, clipped to
+/-5 deg by the bot.

The activation function `f` is **per-neuron**: feature neurons use
`relu_tanh`, internal-state neurons use specialised logic (identity,
cos/sin, gated integration, sample-and-hold), and all remaining neurons
(X19-X999) use `relu_tanh`.


## Input vector `I(t)`

| Index   | Signal                                      |
|---------|---------------------------------------------|
| 0 - 63  | Camera proximity: `1 - depth` (64 rays, 0=leftmost, 63=rightmost) |
| 64      | Hit (1 on collision, 0 otherwise)           |
| 65      | Energy (depletes at 1/1000 per move, 5/1000 on hit) |
| 66      | Bias (always 1.0)                           |


## Neuron map

### Feature neurons X0-X5 (relu_tanh, drive output)

All six are fed through `relu_tanh` and are the **only** neurons with
non-zero `Wout` weights.

| Neuron | Win weights | Role | Wout weight |
|--------|-------------|------|-------------|
| X0 | `Win[0, 64] = 1` | Hit signal | `hit_turn = rad(-10)/tanh(1)` |
| X1 | `Win[1, 20] = 1` | Left-front proximity (ray 20) | `heading_gain = rad(-40)` |
| X2 | `Win[2, 43] = 1` | Right-front proximity (ray 43) | `-heading_gain = rad(+40)` |
| X3 | `Win[3, 11] = -1`, `Win[3, 66] = 0.75` | Left safety (ray 11 - target) | `safety_gain_left = rad(-20)` |
| X4 | `Win[4, 52] = -1`, `Win[4, 66] = 0.75` | Right safety (ray 52 - target) | `safety_gain_right = rad(+20)` |
| X5 | `Win[5, 31] = 1`, `Win[5, 32] = 1`, `Win[5, 66] = -1.4` | Front-block gate (rays 31+32 - threshold) | `front_gain = rad(-20)` |

#### Steering equation

```
O = hit_turn * X0
  + heading_gain * X1  +  (-heading_gain) * X2
  + safety_gain_left * X3  +  safety_gain_right * X4
  + front_gain * X5
```

Clipped to +/-5 deg by the bot.  Negative = clockwise (right turn).

- **Heading symmetry** (X1, X2): steers to equalise left/right front
  proximity.  During straight wall following, the left ray (20) sees the
  outer wall and the right ray (43) sees open space, producing a mild
  clockwise bias.
- **Safety valves** (X3, X4): fire when a side ray deviates from the
  target proximity (0.75), nudging the bot away from the wall.
- **Front-block gate** (X5): fires when the two true-centre rays detect
  a close obstacle (sum > 1.4), commanding a strong clockwise turn into
  corners.
- **Hit snap** (X0): on collision, commands a large clockwise turn that
  saturates the +/-5 deg clamp.


### Head-direction & position tracking X6-X14 (custom activation, zero Wout)

These neurons internally estimate the bot's heading and (x, y) position
using dead reckoning.  They are computed inside the custom activation
function `make_activation` with zero-lag corrections.

| Neuron | Type | Win / W weights | Description |
|--------|------|-----------------|-------------|
| X6 | Computed in `f` | -- | Current clamped dtheta: `clip(Wout[0,:6] @ relu_tanh(X0..X5), -5deg, +5deg)`. Stored so X7 can accumulate it next step. |
| X7 | Identity | `W[7,7]=1, W[7,6]=1` | Direction accumulator: `X7(t+1) = X7(t) + X6(t)`. Cumulative sum of past dtheta with 1-step lag. The activation adds `dtheta_now` on the fly for zero-lag direction. |
| X8 | Computed in `f` | -- | `cos(theta_model)` where `theta_model = X7 + pi/2 + correction + dtheta_now` (zero-lag). |
| X9 | Computed in `f` | -- | `sin(theta_model)` (zero-lag). |
| X10 | Gated integrator | `W[10,10]=1` | x-displacement: `X10(t+1) = X10(t) + speed * cos(theta)` when not hitting, frozen on hit. `model_x = 0.5 + X10`. |
| X11 | Gated integrator | `W[11,11]=1` | y-displacement: same as X10 for the y-axis. `model_y = 0.5 + X11`. |
| X12 | relu_tanh | `Win[12, 64]=1` | Hit signal copy (used to gate X10/X11 position integration). |
| X13 | Sample-and-hold | `W[13,13]=1`, `Win[13,20]=-cal`, `Win[13,43]=+cal` | Latched heading correction. At step 0: captures `cal * (prox_R - prox_L)` as an estimate of the unknown +/-5 deg initial noise. Held via self-recurrence thereafter. `cal = 1/0.173` (empirically calibrated). |
| X14 | Identity (no recurrence) | `Win[14,20]=-cal`, `Win[14,43]=+cal` | Instantaneous sensor asymmetry. Same Win as X13 but no self-loop, so it always reflects the current step. `X13 - X14` recovers the latched step-0 correction at all subsequent steps. |

#### Direction estimation detail

At each step the activation function computes:

```
dtheta_now  = clip(Wout_features @ relu_tanh(X0..X5), -5deg, +5deg)
val7        = X7(t)                        # lagged cumulative dtheta
correction  = X13 - X14                    # latched step-0 asymmetry
              (at step 0: correction = X13, since X14 has same value)
theta_model = val7 + pi/2 + correction + dtheta_now
```

This yields zero-lag direction because `dtheta_now` is added
immediately rather than waiting for X7 to accumulate it.

#### Position estimation detail

```
if hit < 0.5:
    X10(t+1) = X10(t) + speed * cos(theta_model)
    X11(t+1) = X11(t) + speed * sin(theta_model)
else:
    X10(t+1) = X10(t)       # freeze on collision
    X11(t+1) = X11(t)
```

Position in world coordinates: `(0.5 + X10, 0.5 + X11)`.


### Reward-state circuit X15-X18 (relu_tanh, zero Wout)

Detects energy refill events and latches a persistent `is_rewarded`
flag.  These neurons operate entirely through `relu_tanh` applied by
the standard fallthrough in the activation function (no special
per-neuron logic needed).

| Neuron | Win / W weights | Description |
|--------|-----------------|-------------|
| X15 | `Win[15, 65] = K = 0.005` | Delayed energy copy.  Stores a scaled version of the previous step's energy in the near-linear regime of `relu_tanh`.  The scaling factor K = 0.005 keeps `K * energy` small enough that `tanh(K*e) ~ K*e`. |
| X16 | `W[16,15] = 1000`, `W[16,16] = 10` | Armed latch.  Becomes high once X15 has stored one valid energy sample (i.e. after step 0).  Self-recurrence keeps it pinned high.  Gates X17 off during the uninitialised startup step. |
| X17 | `Win[17,65] = 500`, `W[17,15] = -100000`, `W[17,16] = 1000`, `Win[17,66] = -1000.2` | Transient reward pulse.  Fires for exactly one step when `energy(t) > energy(t-1)` (net increase) **and** the arm (X16) is high.  Logic: `relu_tanh(500*e(t) - 100000*X15(t) + 1000*X16(t) - 1000.2)`.  The large `pulse_gain = 100000` amplifies the energy difference; the bias `-1000.2` cancels the arm gate plus a threshold of 0.2 to suppress noise. |
| X18 | `W[18,17] = 10`, `W[18,18] = 10` | Latched is_rewarded flag.  Self-exciting latch driven by the X17 pulse.  Once triggered: `relu_tanh(10*pulse + 10*X18) ~ 1`, then `relu_tanh(0 + 10*1) ~ 1` indefinitely. |

#### Reward detection timeline

| Step | X15 | X16 | X17 | X18 | Notes |
|------|-----|-----|-----|-----|-------|
| 0 | 0 | 0 | 0 | 0 | No prior energy stored; arm is low |
| 1 | K*e(0) | ~1 (armed) | 0 | 0 | X16 latches high from X15; X17 blocked by bias since no energy jump |
| ... | K*e(t-1) | ~1 | 0 | 0 | Normal operation, energy slowly decreasing |
| t_reward | K*e(t-1) | ~1 | **pulse** | 0 | Energy jumped up (refill); X17 fires |
| t_reward+1 | K*e(t) | ~1 | 0 | **~1** | X18 latches from pulse; X17 returns to 0 |
| t_reward+2.. | K*e(t-1) | ~1 | 0 | ~1 | X18 stays pinned via self-recurrence |


## Weight matrices summary

### Win (n x 67) -- non-zero entries only

| Row | Col | Value | Purpose |
|-----|-----|-------|---------|
| 0 | 64 | 1.0 | X0: hit |
| 1 | 20 | 1.0 | X1: left-front ray |
| 2 | 43 | 1.0 | X2: right-front ray |
| 3 | 11 | -1.0 | X3: left safety ray (negated) |
| 3 | 66 | 0.75 | X3: safety target bias |
| 4 | 52 | -1.0 | X4: right safety ray (negated) |
| 4 | 66 | 0.75 | X4: safety target bias |
| 5 | 31 | 1.0 | X5: centre ray 1 |
| 5 | 32 | 1.0 | X5: centre ray 2 |
| 5 | 66 | -1.4 | X5: front threshold bias |
| 12 | 64 | 1.0 | X12: hit (for position gating) |
| 13 | 20 | -5.78 | X13: -cal * prox_L |
| 13 | 43 | +5.78 | X13: +cal * prox_R |
| 14 | 20 | -5.78 | X14: -cal * prox_L |
| 14 | 43 | +5.78 | X14: +cal * prox_R |
| 15 | 65 | 0.005 | X15: scaled energy |
| 17 | 65 | 500.0 | X17: amplified energy |
| 17 | 66 | -1000.2 | X17: arm gate + threshold bias |

### W (n x n) -- non-zero entries only

| Row | Col | Value | Purpose |
|-----|-----|-------|---------|
| 7 | 6 | 1.0 | X7 accumulates X6 |
| 7 | 7 | 1.0 | X7 self-recurrence |
| 10 | 10 | 1.0 | X10 position self-recurrence |
| 11 | 11 | 1.0 | X11 position self-recurrence |
| 13 | 13 | 1.0 | X13 correction latch |
| 16 | 15 | 1000.0 | X16 armed from X15 |
| 16 | 16 | 10.0 | X16 self-recurrence (latch) |
| 17 | 15 | -100000.0 | X17 subtract delayed energy |
| 17 | 16 | 1000.0 | X17 arm gate |
| 18 | 17 | 10.0 | X18 driven by X17 pulse |
| 18 | 18 | 10.0 | X18 self-recurrence (latch) |

### Wout (1 x n) -- non-zero entries only

| Col | Value | Neuron |
|-----|-------|--------|
| 0 | rad(-10)/tanh(1) ~ -0.229 | X0: hit turn |
| 1 | rad(-40) ~ -0.698 | X1: heading (left) |
| 2 | rad(+40) ~ +0.698 | X2: heading (right) |
| 3 | rad(-20) ~ -0.349 | X3: safety left |
| 4 | rad(+20) ~ +0.349 | X4: safety right |
| 5 | rad(-20) ~ -0.349 | X5: front-block |


## Activation function per neuron

| Neurons | Activation | Notes |
|---------|------------|-------|
| X0-X5 | `relu_tanh` | Feature extraction |
| X6 | Custom | Overwritten: `clip(Wout_features @ out[:6], -5deg, +5deg)` |
| X7 | Identity | `out[7] = x[7]` (pass-through of accumulated value) |
| X8-X9 | Custom | Overwritten: `cos/sin(theta_model)` |
| X10-X11 | Custom | Gated integrator (freeze on hit) |
| X12 | `relu_tanh` | Hit signal for gating |
| X13 | Custom | Sample-and-hold (step 0 latch / subsequent hold) |
| X14 | Identity | `out[14] = x[14]` (pass-through) |
| X15-X18 | `relu_tanh` | Reward circuit (standard dynamics) |
| X19-X999 | `relu_tanh` | Unused (all weights zero, output zero) |


## Constants

| Name | Value | Description |
|------|-------|-------------|
| n | 1000 | Total neurons |
| p | 64 | Camera resolution (rays) |
| speed | 0.01 | Bot speed (units/step) |
| leak | 1.0 | State leak (full replacement) |
| warmup | 0 | No warmup steps |
| L_idx | 20 | Left-front ray index |
| R_idx | 43 | Right-front ray index (63 - 20) |
| left_side_idx | 11 | Left safety ray |
| right_side_idx | 52 | Right safety ray (63 - 11) |
| C1_idx, C2_idx | 31, 32 | True-centre rays |
| front_thr | 1.4 | Front-block threshold |
| safety_target | 0.75 | Safety valve target proximity |
| cal_gain | 1/0.173 ~ 5.78 | Heading calibration gain |
| K | 0.005 | Energy scaling factor |
| pulse_gain | 100000 | Energy-jump amplification |
| pulse_thr | 0.2 | Pulse threshold (suppresses noise) |
| arm_gate | 1000 | Arm gate strength |
| arm_from_energy | 1000 | Arm latch drive from energy |
| arm_latch | 10 | Arm self-recurrence |
| latch_gain | 10 | Is_rewarded self-excitation |


## Circuit diagram (signal flow)

```
INPUTS                    FEATURE NEURONS              OUTPUT
-----------               ---------------              ------
ray[20] -----> X1 (L prox) ---+
ray[43] -----> X2 (R prox) ---+
ray[11] -----> X3 (safety L) -+--> Wout --> O (steering, +/-5 deg)
ray[52] -----> X4 (safety R) -+
ray[31,32] --> X5 (front)  ---+
hit ---------> X0 (hit)    ---+

                    HEAD-DIRECTION TRACKING (zero Wout)
                    -----------------------------------
               X0..X5 --> dtheta_now (X6)
                                |
                  X6 --> X7 (accumulator: sum of past dtheta)
                                |
          ray[20,43] --> X13 (sample-and-hold correction)
          ray[20,43] --> X14 (instantaneous correction)
                                |
                  X7 + correction + dtheta_now + pi/2 = theta_model
                                |
                       X8 = cos(theta), X9 = sin(theta)
                                |
              hit --> X12 ----> gate
                                |
                  X10 += speed * cos  (if no hit)
                  X11 += speed * sin  (if no hit)

                    REWARD-STATE DETECTION (zero Wout)
                    ----------------------------------
          energy --> X15 (delayed scaled copy)
                       |
                  X15 --> X16 (armed latch, self-recurrent)
                       |
          energy --> X17 (pulse: energy_now - energy_prev, gated by X16)
                       |
                  X17 --> X18 (is_rewarded latch, self-recurrent)
```


## Current status

This is the initial version.  The internal state neurons (X6-X18) are
fully functional but do not influence the output.  The steering behavior
is identical to dummy2/dummy4.  Future versions may use X8-X11 (position
and direction estimates) and X18 (reward flag) to condition the output.
