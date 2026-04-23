# Bio Player — Environment 1

## 1. Overview

`env1_player_bio.py` is a pointwise-activation Echo State Network controller
for Environment 1 of the BrainCraft challenge. Every hidden activation is a
scalar function of its own preactivation; all cross-neuron logic is expressed
by the connectivity matrices. The network runs

```text
X(t+1) = f(Win @ I(t) + W @ X(t))
O(t+1) = Wout @ g(X(t+1))
```

with identity readout `g(x) = x`. The model is produced by a single `yield`
in `bio_player()` (no iterative training), so the matrices are fixed at
build time.

The controller combines three behaviours: a reflex wall-follower, a
heading-gated corridor shortcut driven by pose alone (no colour cues), and
a rising-edge energy-reward detector that arms the shortcut trigger.

This is the env1 adaptation of `env2_player_bio.py`. Relative to the env2
version:

1. **Colour-evidence circuit removed.** Env1 has no colour channel, so the
   per-ray `xi_blue[0..63]` detectors, the half-sum evidence channels
   (`l_ev`, `r_ev`), the dominance pulses (`dleft`, `dright`), the signed
   integrator (`evidence`), the evidence thresholds (`trig_pos`,
   `trig_neg`), and the latched front-sign neurons (`fs_pos`, `fs_neg`) are
   all deleted.
2. **Single unsigned front-block channel.** Without a latched colour sign,
   the two env2 channels `front_block_pos` / `front_block_neg` collapse to a
   single `front_block` neuron with a fixed-direction readout. The bot
   always turns the same way (CCW, `+front_gain_mag = +20°`) when a
   centre-front obstacle is detected.
3. **Re-indexed inputs.** The env2 input vector
   `[prox, colour, hit, energy, bias]` of length `2p + 3 = 131` is replaced
   by the env1 vector `[prox, hit, energy, bias]` of length `p + 3 = 67`,
   with `hit_idx = p`, `energy_idx = p + 1`, `bias_idx = p + 2`.
4. **Initial-heading correction circuit removed.** The env2 `seed_pos`,
   `seed_neg`, `head_corr`, and `init_impulse` slots are dropped. Env1 is
   robust to the ±5° start perturbation because the reflex wall-follower
   re-aligns the bot long before pos_x drift reaches the shortcut corridor
   bumps (half-width `near_c_thr = 0.05`); a 500-seed sweep confirms the
   simplification does not regress the score distribution (see §7). The
   `step_counter` + `armed_flag` pair is kept as a minimal reward arm
   gate so the detector is held off during the first `arm_window_k = 6`
   steps.

The heading-gated shortcut trigger, the reward latch, the quadrant-AND
turn steering, and the `is_app`-gated reflex suppression carry over from
env2 unchanged.

## 2. Network shape

| Parameter     | Value                          |
| ------------- | ------------------------------ |
| `n`           | `1000`                         |
| `p`           | `64` (camera rays)             |
| `n_inputs`    | `p + 3 = 67`                   |
| `warmup`      | `0`                            |
| `leak` (λ)    | `1.0`                          |
| `g`           | identity                       |
| Actuator clip | `step_a = 5°`                  |
| Bot speed     | `0.01` (from `bot.speed`)      |

## 3. Activation library

Each neuron `i` has a fixed scalar activation applied pointwise to its
preactivation `z_i(t)`:

| Name        | Formula                       | Used by                                                                      |
| ----------- | ----------------------------- | ---------------------------------------------------------------------------- |
| `relu_tanh` | `max(0, tanh(z))`             | default — threshold, latch, AND/OR gates                                     |
| `identity`  | `z`                           | `dir_accum`, `pos_x`, `pos_y`, `shortcut_steer`, `step_counter`              |
| `relu`      | `max(0, z)`                   | `energy_ramp`, `sc_countdown`                                                |
| `clip_a`    | `clip(z, -step_a, +step_a)`   | `dtheta`                                                                     |
| `sin`       | `sin(z)`                      | `sin_n`, `cos_n`                                                             |
| `bump`      | `max(0, 1 - 4 z^2)`           | `near_e`, `near_w`                                                           |

Module constants:

```text
shortcut_turn  = -2.0
near_c_thr     = 0.05
drift_offset   = 0.175
turn_steps     = 18
approach_steps = 50
sc_total       = 68
front_gain_mag = 20°
k_sharp        = 50.0
ncr_gain       = 2.5
near_cr_gain   = 2.5
step_a         = 5°
arm_window_k   = 6
```

## 4. Inputs and slot layout

The input vector is

```text
I(t) = [prox[0..63](t), hit(t), energy(t), 1]
```

with these index aliases used by the controller:

```text
L_idx          = 20        (left reflex proximity tap)
R_idx          = 43        (right reflex proximity tap)
left_side_idx  = 11        (left safety tap)
right_side_idx = 52        (right safety tap)
C1_idx, C2_idx = 31, 32    (two centre-front proximity taps)
hit_idx        = 64
energy_idx     = 65
bias_idx       = 66
```

Hidden slots in allocator order (`n = 1000`; slots `36..999` receive no
incoming weights):

| Slot | Name              | Activation  | Role                                               |
| ---- | ----------------- | ----------- | -------------------------------------------------- |
| 0    | `hit_feat`        | `relu_tanh` | hit reflex                                         |
| 1    | `prox_left`       | `relu_tanh` | left proximity reflex                              |
| 2    | `prox_right`     | `relu_tanh` | right proximity reflex                             |
| 3    | `safe_left`       | `relu_tanh` | left safety feature                                |
| 4    | `safe_right`      | `relu_tanh` | right safety feature                               |
| 5    | `dtheta`          | `clip_a`    | clipped one-step-lagged steering command           |
| 6    | `dir_accum`       | `identity`  | integrated heading                                 |
| 7    | `pos_x`           | `identity`  | integrated x position                              |
| 8    | `pos_y`           | `identity`  | integrated y position                              |
| 9    | `step_counter`    | `identity`  | step index (drives `armed_flag` timing)            |
| 10   | `armed_flag`      | `relu_tanh` | reward arm-gate latch                              |
| 11   | `energy_ramp`     | `relu`      | previous-step energy, for rising-edge detection    |
| 12   | `reward_pulse`    | `relu_tanh` | energy rising-edge detector                        |
| 13   | `reward_latch`    | `relu_tanh` | latched reward-seen signal                         |
| 14   | `sc_countdown`    | `relu`      | shortcut phase countdown                           |
| 15   | `shortcut_steer`  | `identity`  | shortcut steering actuator                         |
| 16   | `sin_n`           | `sin`       | `sin(phi)`                                         |
| 17   | `cos_n`           | `sin`       | `cos(phi)` (via `sin(phi + π/2)`)                  |
| 18   | `sin_pos`         | `relu_tanh` | `sin(phi) > 0` sharp detector                      |
| 19   | `sin_neg`         | `relu_tanh` | `sin(phi) < 0` sharp detector                      |
| 20   | `y_pos`           | `relu_tanh` | `pos_y > 0` sharp detector                         |
| 21   | `y_neg`           | `relu_tanh` | `pos_y < 0` sharp detector                         |
| 22   | `near_e`          | `bump`      | east-shifted corridor bump                         |
| 23   | `near_w`          | `bump`      | west-shifted corridor bump                         |
| 24   | `near_cr_e`       | `relu_tanh` | AND(`near_e`, heading east)                        |
| 25   | `near_cr_w`       | `relu_tanh` | AND(`near_w`, heading west)                        |
| 26   | `near_cr`         | `relu_tanh` | corridor predicate (OR of `near_cr_e`, `near_cr_w`)|
| 27   | `trig_sc`         | `relu_tanh` | shortcut trigger pulse                             |
| 28   | `on_countdown`    | `relu_tanh` | `sc_countdown > 0.5`                               |
| 29   | `is_turn`         | `relu_tanh` | turn-phase gate                                    |
| 30   | `is_app`          | `relu_tanh` | approach-phase gate                                |
| 31   | `sy_pp`           | `relu_tanh` | AND(`sin_pos`, `y_pos`, `is_turn`)                 |
| 32   | `sy_pn`           | `relu_tanh` | AND(`sin_pos`, `y_neg`, `is_turn`)                 |
| 33   | `sy_np`           | `relu_tanh` | AND(`sin_neg`, `y_pos`, `is_turn`)                 |
| 34   | `sy_nn`           | `relu_tanh` | AND(`sin_neg`, `y_neg`, `is_turn`)                 |
| 35   | `front_block`     | `relu_tanh` | unsigned front-block detector                      |

## 5. Main circuits

### 5.1 Reflex features and readout

The reflex channels are feed-forward proximity/hit detectors suppressed
during the shortcut approach phase:

```text
hit_feat(t+1)   = relu_tanh(hit(t)                       - k_sharp * is_app(t))
prox_left(t+1)  = relu_tanh(prox[L_idx](t)               - k_sharp * is_app(t))
prox_right(t+1) = relu_tanh(prox[R_idx](t)               - k_sharp * is_app(t))
safe_left(t+1)  = relu_tanh(-prox[left_side_idx](t)  + 0.75 - k_sharp * is_app(t))
safe_right(t+1) = relu_tanh(-prox[right_side_idx](t) + 0.75 - k_sharp * is_app(t))
```

The steering readout is

```text
O(t+1) =
    hit_turn          * hit_feat(t+1)
  + heading_gain      * prox_left(t+1)
  - heading_gain      * prox_right(t+1)
  + safety_gain_left  * safe_left(t+1)
  + safety_gain_right * safe_right(t+1)
  + front_gain_mag    * front_block(t+1)
  + shortcut_steer(t+1)
```

with

```text
hit_turn          = -10° / tanh(1)
heading_gain      = -40°
safety_gain_left  = -20°
safety_gain_right = +20°
front_gain_mag    = +20°
```

The `dtheta` slot holds the clipped one-step-lagged steering command:

```text
dtheta(t+1) = clip(O(t), -step_a, +step_a)
```

implemented by mirroring `Wout` row 0 into `W[dtheta, :]`.

### 5.2 Heading, trig, and position

```text
dir_accum(t+1) = dir_accum(t) + dtheta(t)

phi(t)      = dir_accum(t) + dtheta(t)
sin_n(t+1)  = sin(phi(t))
cos_n(t+1)  = sin(phi(t) + π/2)       # = cos(phi(t))

pos_x(t+1) = pos_x(t) - speed * sin_n(t)
pos_y(t+1) = pos_y(t) + speed * cos_n(t)
```

The sin-only trig pair keeps the activation library minimal while still
covering the environment frame `dx = -speed·sin(phi)`,
`dy = +speed·cos(phi)` used by the shortcut predicates. Here `phi` is the
controller's internal heading measured relative to north, i.e. `phi = 0`
means north, `phi = -π/2` means east, and `phi = +π/2` means west.

Since the initial-heading correction circuit is dropped, the controller's
internal `phi` is offset from the bot's true heading by the random
start-direction perturbation (±5°). The downstream trigger tolerates this:
`near_cr_e` / `near_cr_w` gate on `|sin_n|` within ~±60° of horizontal,
and `pos_x` drifts by at most a few millimetres over the steps needed to
reach a corridor — well inside the `±near_c_thr = ±0.05` bump width.

### 5.3 Reward arm gate and reward circuit

With `pulse_gain = 500`, `pulse_thr = 0.2`, `arm_gate = 1000`,
`latch_gain = 10`, `arm_window_k = 6`:

```text
step_counter(t+1) = step_counter(t) + 1
armed_flag(t+1)   = relu_tanh(k_sharp * (step_counter(t) - (arm_window_k - 1.5)))

energy_ramp(t+1)  = relu(energy(t))

reward_pulse(t+1) = relu_tanh(
    pulse_gain * energy(t)
  - pulse_gain * energy_ramp(t)
  + arm_gate   * armed_flag(t)
  - (arm_gate + pulse_thr)
)

reward_latch(t+1) = relu_tanh(
    latch_gain * reward_pulse(t)
  + latch_gain * reward_latch(t)
)
```

`armed_flag(t) = 0` for `t = 0..5` and saturates to `1` for `t ≥ 6`. The
`-(arm_gate + pulse_thr)` bias holds the pulse off while `armed_flag` is
zero, and once it saturates the node reduces to a sharp rising-edge
detector on `energy(t) - energy(t-1)`. The bot cannot reach any source
inside the six-step window, so arming only after it closes is safe.

### 5.4 Corridor tests and shortcut trigger

Two bump corridor detectors read `pos_x`:

```text
near_e(t+1) = bump((pos_x(t) + drift_offset) / (2*near_c_thr))
near_w(t+1) = bump((pos_x(t) - drift_offset) / (2*near_c_thr))
```

`near_e` peaks when the bot is near `pos_x = -drift_offset` (west of
centre, approaching the corridor from the east-bound leg); `near_w` is
the mirror. Each is combined with a heading check into a 2-way AND.
Heading is read from `sin_n`: since the internal `phi` is measured
relative to north, `sin_n = sin(phi)` equals `-1` when the bot is
heading east and `+1` when heading west.

```text
near_cr_e(t+1) = relu_tanh(ncr_gain * k_sharp * (near_e(t) - sin_n(t) - 1.5))
near_cr_w(t+1) = relu_tanh(ncr_gain * k_sharp * (near_w(t) + sin_n(t) - 1.5))
```

The ±0.5 margin in each AND accepts headings within ~±60° of
horizontal (the normal perimeter approach has `|sin_n| > 0.95` at the
trigger moment) while rejecting perpendicular crossings of
`pos_x = ±drift_offset` on later laps. The `ncr_gain = 2.5` sharpens
the AND so near-threshold inputs don't fire the gate partially.

The corridor predicate is the OR of the two heading-gated detectors,
with a sharpened OR gate (`near_cr_gain = 2.5`) so that small roundoff
in the inputs cannot produce an intermediate `near_cr`:

```text
near_cr(t+1) = relu_tanh(near_cr_gain * k_sharp * (near_cr_e(t) + near_cr_w(t) - 0.5))
```

The shortcut trigger is a 2-way AND with two refractory terms, since
position-at-corridor and heading-aligned-with-corridor are both already
captured by `near_cr`:

```text
trig_sc(t+1) = relu_tanh(
    k_sharp * (reward_latch(t) + near_cr(t) - 1.5)
  - 10 * k_sharp * trig_sc(t)
  -      k_sharp * sc_countdown(t)
)
```

### 5.5 Shortcut countdown, phases, and steering

```text
sc_countdown(t+1) = relu(sc_countdown(t) - 1 + (sc_total + 1) * trig_sc(t))

on_countdown(t+1) = relu_tanh(k_sharp * (sc_countdown(t) - 0.5))
is_turn(t+1)      = relu_tanh(k_sharp * (sc_countdown(t) - (approach_steps + 0.5)))
is_app(t+1)       = relu_tanh(k_sharp * (on_countdown(t) - is_turn(t) - 0.5))
```

During the turn phase, four quadrant ANDs implement
`turn_toward = sign(sin(phi)) · sign(pos_y)`:

```text
sy_pp(t+1) = relu_tanh(k_sharp * (sin_pos(t) + y_pos(t) + is_turn(t) - 2.5))
sy_pn(t+1) = relu_tanh(k_sharp * (sin_pos(t) + y_neg(t) + is_turn(t) - 2.5))
sy_np(t+1) = relu_tanh(k_sharp * (sin_neg(t) + y_pos(t) + is_turn(t) - 2.5))
sy_nn(t+1) = relu_tanh(k_sharp * (sin_neg(t) + y_neg(t) + is_turn(t) - 2.5))

shortcut_steer(t+1) = |shortcut_turn| * (sy_pp(t) + sy_nn(t) - sy_pn(t) - sy_np(t))
```

### 5.6 Front block (unsigned)

Env1 has no colour input, so the signed `front_block_pos` /
`front_block_neg` pair of the env2 player collapses to a single
unsigned detector driven by the two centre proximity taps:

```text
front_block(t+1) = relu_tanh(C1(t) + C2(t) - front_thr)
```

with `front_thr = 1.4`. A positive reading turns the bot by
`+front_gain_mag = +20°` via the readout — a fixed-direction
"bounce-off-the-wall" escape. The rotation direction is arbitrary
(without colour cues there is no preferred side); we stick to CCW.

## 6. Nonzero readout weights

```text
Wout[hit_feat]        = hit_turn          = -10° / tanh(1)
Wout[prox_left]       = heading_gain      = -40°
Wout[prox_right]      = -heading_gain     = +40°
Wout[safe_left]       = safety_gain_left  = -20°
Wout[safe_right]      = safety_gain_right = +20°
Wout[front_block]     = +front_gain_mag   = +20°
Wout[shortcut_steer]  = +1
```

All other `Wout` entries are zero (seven nonzero weights total). The
controller also mirrors this row into `W[dtheta, :]`, so `dtheta(t+1)`
is the clipped copy of the previous step's output.

## 7. Verification

```bash
python braincraft/env1_player_bio.py
```

Runs `train(bio_player, timeout=100)` followed by
`evaluate(model, Bot, Environment, debug=False, seed=12345)` on 10
runs and prints the mean distance and standard deviation.

Observed current output is:

```text
Final score (distance): 14.41 +/- 0.36
```

For a thorough per-seed distribution:

```bash
python braincraft/validate_env1_player_bio.py --n-seeds 500 --workers 24
```

Representative 500-seed × 10-episode sweep (5000 episodes, numpy 1.26.4):

```text
Across-seed-mean  mean = 14.3795
Across-seed-mean  std  = 0.1314
Across-seed-mean  min  = 14.0550
Across-seed-mean  max  = 14.8480

Within-seed std   mean = 0.3706

Per-seed-mean quantiles
  q=0.01  14.0750    q=0.50  14.3875    q=0.90  14.5400
  q=0.05  14.1669    q=0.75  14.4672    q=0.99  14.6821

Catastrophic-failure tail (per-seed mean below threshold)
  seeds with mean <  12.00 / 13.00 / 13.50 / 14.00 :  0 / 0 / 0 / 0
  seeds with mean <  14.50 / 14.60 / 14.70         :  417 / 474 / 496

Per-episode tail (5000 episodes)
  episodes <  12.00 / 13.00: 0 / 0
  episodes <  14.00 / 14.50: 2415 / 2666
```

No seed drops below `14.00`, the tightest-tail threshold that still
separates the shortcut-taking regime from a plain wall-follower. The
worst seed yields `14.05` (seed `173`) and the best `14.85` (seed
`497`); the median is `14.39` with across-seed std `0.13`.

Sanity invariants for the built matrices:

- `Win.shape == (1000, 67)` and `W.shape == (1000, 1000)`.
- Only slots `0..35` receive any incoming weights.
- `Wout` has exactly seven nonzero entries (listed in §6).
