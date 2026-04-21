# Reflex Bio Player 2 - Environment 2

## 1. Overview

`env2_player_reflex_bio2.py` is the pointwise-activation version of the env2
bio controller. After construction, all runtime logic is carried by fixed
scalar activations plus weights in `Win` and `W`:

```text
X(t+1) = f(Win @ I(t) + W @ X(t))
O(t+1) = Wout @ g(X(t+1))
```

with identity readout `g(x) = x`.

The concrete build uses:

- `n = 1000` hidden units
- `p = 64` camera rays
- `n_inputs = 2*p + 3 = 131`
- `warmup = 0`
- `leak = 1`
- actuator clip `step_a = 5 deg`
- bot step `speed = 0.01` (from `bot.speed`, used by `pos_x` / `pos_y`)

The controller keeps the accepted env2 behavior while expressing the reward
logic, shortcut state machine, trig helpers, and position tracking as fixed
neurons with fixed activations.

## 2. Activation Library

For each neuron preactivation `z_i(t)`, the hidden update applies a fixed
scalar function `f_i`:

| Name | Formula | Used by |
| --- | --- | --- |
| `relu_tanh` | `max(0, tanh(z))` | default thresholds, gates, latches |
| `identity` | `z` | `dir_accum`, `pos_x`, `pos_y`, `head_corr`, `shortcut_steer`, `init_impulse`, `evidence`, `l_ev`, `r_ev`, `step_counter` |
| `relu` | `max(0, z)` | `energy_ramp`, `sc_countdown` |
| `clip_a` | `clip(z, -step_a, step_a)` | `dtheta` |
| `sin` | `sin(z)` | `cos_n`, `sin_n` |
| `square` | `z^2` | `sin_sq` |
| `bump` | `max(0, 1 - 4 z^2)` | `near_e`, `near_w`, `xi_blue[*]` |

Module constants:

```text
shortcut_turn      = -2.0
sin_horiz_thr      = 0.35
near_c_thr         = 0.05
drift_offset       = 0.175
turn_steps         = 18
approach_steps     = 45
sc_total           = 63
color_evidence_thr = 2.0
front_gain_mag     = 20 deg
gate_c             = 1.0
k_sharp            = 50.0
step_a             = 5 deg
cal_gain           = 1 / 0.173
seed_window_k      = 6
```

## 3. Inputs and Allocator

The input vector is

```text
I(t) = [prox[0..63](t), color[0..63](t), hit(t), energy(t), 1].
```

Ray aliases used by the controller:

```text
L_idx          = 20
R_idx          = 43
left_side_idx  = 11
right_side_idx = 52
C1_idx, C2_idx = 31, 32
hit_idx        = 128
energy_idx     = 129
bias_idx       = 130
```

Live bio2 slots in allocator order:

| Slot | Key | Activation | Role |
| --- | --- | --- | --- |
| `0` | `hit_feat` | `relu_tanh` | hit reflex feature |
| `1` | `prox_left` | `relu_tanh` | left proximity reflex feature |
| `2` | `prox_right` | `relu_tanh` | right proximity reflex feature |
| `3` | `safe_left` | `relu_tanh` | left safety feature |
| `4` | `safe_right` | `relu_tanh` | right safety feature |
| `5` | `dtheta` | `clip_a` | clipped delayed steering state |
| `6` | `dir_accum` | `identity` | integrated heading state |
| `7` | `pos_x` | `identity` | integrated x-like corridor coordinate |
| `8` | `pos_y` | `identity` | integrated y-like corridor coordinate |
| `9` | `head_corr` | `identity` | latched initial heading correction |
| `10` | `seeded_flag` | `relu_tanh` | one-way seed latch |
| `11` | `step_counter` | `identity` | step index (0, 1, 2, ...), drives `seeded_flag` timing |
| `12` | `seed_pos` | `relu_tanh` | positive initial-correction pulse |
| `13` | `seed_neg` | `relu_tanh` | negative initial-correction pulse |
| `14` | `energy_ramp` | `relu` | energy ramp for reward trigger |
| `15` | `reward_pulse` | `relu_tanh` | reward pulse detector |
| `16` | `reward_latch` | `relu_tanh` | latched reward signal |
| `17` | `sc_countdown` | `relu` | shortcut countdown |
| `18` | `shortcut_steer` | `identity` | shortcut steering term |
| `19` | `init_impulse` | `identity` | one-step initial correction pulse |
| `20` | `sin_n` | `sin` | `sin(phi)` trig helper for `pos_x` |
| `21` | `cos_n` | `sin` | `cos(phi)` trig helper for `pos_y` |
| `22` | `sin_sq` | `square` | `cos_n^2` magnitude (legacy name) |
| `23` | `cos_pos` | `relu_tanh` | `-sin(phi) > 0` detector (legacy quadrant name) |
| `24` | `cos_neg` | `relu_tanh` | `-sin(phi) < 0` detector (legacy quadrant name) |
| `25` | `y_pos` | `relu_tanh` | `pos_y > 0` detector |
| `26` | `y_neg` | `relu_tanh` | `pos_y < 0` detector |
| `27` | `near_e` | `bump` | east-shifted corridor detector |
| `28` | `near_w` | `bump` | west-shifted corridor detector |
| `29` | `near_cr` | `relu_tanh` | corridor predicate (OR of `near_e`, `near_w`) |
| `30` | `heading_horiz` | `relu_tanh` | near-horizontal heading detector |
| `31` | `front_clear` | `relu_tanh` | no front-block detector |
| `32` | `trig_sc` | `relu_tanh` | shortcut trigger pulse |
| `33` | `on_countdown` | `relu_tanh` | `sc_countdown > 0.5` |
| `34` | `is_turn` | `relu_tanh` | turn phase gate |
| `35` | `is_app` | `relu_tanh` | approach phase gate |
| `36` | `cy_pp` | `relu_tanh` | `cos_pos AND y_pos AND is_turn` |
| `37` | `cy_pn` | `relu_tanh` | `cos_pos AND y_neg AND is_turn` |
| `38` | `cy_np` | `relu_tanh` | `cos_neg AND y_pos AND is_turn` |
| `39` | `cy_nn` | `relu_tanh` | `cos_neg AND y_neg AND is_turn` |
| `40` | `front_block_pos` | `relu_tanh` | front block term for positive front sign |
| `41` | `front_block_neg` | `relu_tanh` | front block term for negative front sign |
| `42` | `l_ev` | `identity` | left-half blue evidence sum |
| `43` | `r_ev` | `identity` | right-half blue evidence sum |
| `44` | `dleft` | `relu_tanh` | left-dominance pulse |
| `45` | `dright` | `relu_tanh` | right-dominance pulse |
| `46` | `evidence` | `identity` | signed color evidence accumulator |
| `47` | `trig_pos` | `relu_tanh` | positive evidence trigger |
| `48` | `trig_neg` | `relu_tanh` | negative evidence trigger |
| `49` | `fs_pos` | `relu_tanh` | latched positive front sign |
| `50` | `fs_neg` | `relu_tanh` | latched negative front sign |
| `51..114` | `xi_blue[0..63]` | `bump` | per-ray blue detector centered at color value `4` |

## 4. Main Circuits

### 4.1 Reflex features and readout

The five reflex features are feed-forward proximity/hit channels, suppressed
during shortcut approach:

```text
hit_feat(t+1)   = relu_tanh(hit(t) - k_sharp*is_app(t))
prox_left(t+1)  = relu_tanh(prox[L_idx](t) - k_sharp*is_app(t))
prox_right(t+1) = relu_tanh(prox[R_idx](t) - k_sharp*is_app(t))
safe_left(t+1)  = relu_tanh(-prox[left_side_idx](t)  + 0.75 - k_sharp*is_app(t))
safe_right(t+1) = relu_tanh(-prox[right_side_idx](t) + 0.75 - k_sharp*is_app(t))
```

The steering readout is

```text
O(t+1) =
    hit_turn          * hit_feat(t+1)
  + heading_gain      * prox_left(t+1)
  - heading_gain      * prox_right(t+1)
  + safety_gain_left  * safe_left(t+1)
  + safety_gain_right * safe_right(t+1)
  + front_gain_mag    * front_block_pos(t+1)
  - front_gain_mag    * front_block_neg(t+1)
  + shortcut_steer(t+1)
  + init_impulse(t+1)
```

with

```text
hit_turn          = -10 deg / tanh(1)
heading_gain      = -40 deg
safety_gain_left  = -20 deg
safety_gain_right = +20 deg
```

The internal steering state is the clipped lagged readout:

```text
dtheta(t+1) = clip(O(t), -step_a, +step_a).
```

### 4.2 Direction, trig, and position

```text
dir_accum(t+1) = dir_accum(t) + dtheta(t)

phi(t)      = dir_accum(t) + head_corr(t) + dtheta(t)
sin_n(t+1)  = sin(phi(t))
cos_n(t+1)  = sin(phi(t) + pi/2)   # = cos(phi(t))
sin_sq(t+1) = cos_n(t)^2

pos_x(t+1) = pos_x(t) - speed * sin_n(t)
pos_y(t+1) = pos_y(t) + speed * cos_n(t)
```

The `sin`-only trig pair preserves the established downstream `pos_x / pos_y`
frame used by the shortcut predicates (`dx = -speed * sin(phi)`,
`dy = +speed * cos(phi)`). The `sin_sq` neuron holds `cos(phi)^2` despite its
legacy name; `heading_horiz` uses it against `sin_horiz_thr`.

### 4.3 Initial heading correction

The raw initial correction is

```text
current_corr(t) = (prox[R_idx](t) - prox[L_idx](t)) * cal_gain.
```

The latch is built from `step_counter`, `seeded_flag`, `seed_pos`,
`seed_neg`, and `head_corr`. `step_counter` is a simple identity-activation
counter that increments each step, and `seeded_flag` is a sharp threshold
against it:

```text
step_counter(t+1) = step_counter(t) + 1           # identity, so step_counter(t) = t
seeded_flag(t+1)  = relu_tanh(k_sharp*(step_counter(t) - (seed_window_k - 1.5)))

seed_pos(t+1) = relu_tanh(-cal_gain*prox[L_idx](t) + cal_gain*prox[R_idx](t) - 1000*seeded_flag(t))
seed_neg(t+1) = relu_tanh( cal_gain*prox[L_idx](t) - cal_gain*prox[R_idx](t) - 1000*seeded_flag(t))

head_corr(t+1)    = head_corr(t) + seed_pos(t) - seed_neg(t)
init_impulse(t+1) = -seed_pos(t) + seed_neg(t)
```

With `seed_window_k = 6`, `seeded_flag(t) = 0` for `t = 0..5` and
saturates to `1` for `t >= 6`. Since `seed_pos(t+1)` reads
`seeded_flag(t)`, the seeds fire for six consecutive network steps
(`t = 1..6`). `head_corr` integrates each step's residual depth
asymmetry, and `init_impulse` steers against the remaining
perturbation. This replaces the previous single-step impulse, which was
exact on good draws but produced a ~1.6° residual on rare bad draws
(positive initial-heading perturbation) that accumulated into the 8th
shortcut approach and caused wall contact. The closed-loop multi-step
correction drives the residual toward zero regardless of perturbation
sign. See [plans/reflex_bio2_init_heading_robustness_plan.md](plans/reflex_bio2_init_heading_robustness_plan.md)
for the K sweep and failure-mode analysis.

### 4.4 Reward circuit

With

```text
pulse_gain      = 500
pulse_thr       = 0.2
arm_gate        = 1000
latch_gain      = 10
```

the reward neurons satisfy

```text
energy_ramp(t+1) = relu(energy(t))

reward_pulse(t+1) = relu_tanh(
    pulse_gain * energy(t)
  - pulse_gain * energy_ramp(t)
  + arm_gate   * seeded_flag(t)
  - (arm_gate + pulse_thr)
)

reward_latch(t+1) = relu_tanh(
    latch_gain * reward_pulse(t)
  + latch_gain * reward_latch(t)
)
```

`seeded_flag` (already used to gate the initial-heading seeds) acts as the
arm gate: it is `0` for the first `seed_window_k` steps and saturates to
`1` afterward. During the unseeded window `reward_pulse` is clamped off
by the `-1000.2` bias, and from step `seed_window_k + 1` onward the
neuron reduces to `relu_tanh(500 * Delta energy(t) - 0.2)` -- a sharp
rising-edge detector on the energy signal. Delaying arming by a handful
of steps is safe because the first reward event does not occur until
step ~73 and the bot, starting from a fixed non-source cell, covers too
little ground in `seed_window_k ~ 6` steps to reach any source. This
removes the dedicated `armed_latch` neuron: its sole purpose was to
wait for the first non-zero `energy_ramp` before arming, and
`seeded_flag` provides the same guarantee without extra state.

### 4.5 Corridor tests and shortcut trigger

Two bump-based offset corridor detectors are driven by `pos_x`. `near_e`
fires when `pos_x` is near `-drift_offset` (bot west of centre, about to
cross if heading east); `near_w` is the mirror:

```text
near_e(t+1) = bump((pos_x(t) + drift_offset) / (2*near_c_thr))
near_w(t+1) = bump((pos_x(t) - drift_offset) / (2*near_c_thr))
```

The corridor predicate `near_cr` is simply their OR. An earlier layout
included a central detector `near_c` plus directional gates
(`cos_big_pos`/`cos_big_neg`/`cos_small` feeding `ncr_e`/`ncr_w`/`ncr_c`),
but those seven helpers were provably redundant under the
`heading_horiz` (`|sin(phi)| > 0.937`) conjunction in `trig_sc` — ablation
left the accepted score unchanged, so they were removed:

```text
near_cr(t+1) = relu_tanh(k_sharp*(near_e(t) + near_w(t) - 0.5))

heading_horiz(t+1) = relu_tanh(k_sharp*(1 - sin_sq(t) / sin_horiz_thr^2))
front_clear(t+1)   = relu_tanh(k_sharp*(0.1 - front_block_pos(t) - front_block_neg(t)))
```

The shortcut trigger is a 4-way AND with refractory feedback:

```text
trig_sc(t+1) = relu_tanh(
    k_sharp*(reward_latch(t) + heading_horiz(t) + front_clear(t) + near_cr(t) - 3.5)
  - 10*k_sharp*trig_sc(t)
  - k_sharp*sc_countdown(t)
)
```

### 4.6 Shortcut phases and steering

The countdown and phase gates are

```text
sc_countdown(t+1) = relu(sc_countdown(t) - 1 + (sc_total + 1) * trig_sc(t))

on_countdown(t+1) = relu_tanh(k_sharp*(sc_countdown(t) - 0.5))
is_turn(t+1)      = relu_tanh(k_sharp*(sc_countdown(t) - (approach_steps + 0.5)))
is_app(t+1)       = relu_tanh(k_sharp*(on_countdown(t) - is_turn(t) - 0.5))
```

During the turn phase, four quadrant detectors implement
`turn_toward = sign(sin(phi)) * sign(pos_y)`. The `cos_pos / cos_neg`
helpers keep their legacy names but now test `sign(-sin(phi))`, so the
same wiring yields the same turn direction as before the rename:

```text
cy_pp(t+1) = relu_tanh(k_sharp*(cos_pos(t) + y_pos(t) + is_turn(t) - 2.5))
cy_pn(t+1) = relu_tanh(k_sharp*(cos_pos(t) + y_neg(t) + is_turn(t) - 2.5))
cy_np(t+1) = relu_tanh(k_sharp*(cos_neg(t) + y_pos(t) + is_turn(t) - 2.5))
cy_nn(t+1) = relu_tanh(k_sharp*(cos_neg(t) + y_neg(t) + is_turn(t) - 2.5))

shortcut_steer(t+1) =
    abs(shortcut_turn) * (cy_pn(t) + cy_np(t) - cy_pp(t) - cy_nn(t))
```

### 4.7 Blue evidence and front-block sign

Each blue detector is a bump centered at color value `4`:

```text
xi_blue[r](t+1) = bump(color[r](t) - 4).
```

The left and right evidence sums use the two ray halves:

```text
l_ev(t+1) = sum_{r=0}^{31}  xi_blue[r](t)
r_ev(t+1) = sum_{r=32}^{63} xi_blue[r](t)
```

Dominance pulses and the signed integrator:

```text
dleft(t+1)  = relu_tanh(k_sharp*(l_ev(t) - r_ev(t) - 0.2) - 10*k_sharp*fs_pos(t) - 10*k_sharp*fs_neg(t))
dright(t+1) = relu_tanh(k_sharp*(r_ev(t) - l_ev(t) - 0.2) - 10*k_sharp*fs_pos(t) - 10*k_sharp*fs_neg(t))

evidence(t+1) = evidence(t) + dright(t) - dleft(t)
trig_pos(t+1) = relu_tanh(k_sharp*(evidence(t) - (color_evidence_thr - 0.5)))
trig_neg(t+1) = relu_tanh(k_sharp*(-evidence(t) - (color_evidence_thr - 0.5)))

fs_pos(t+1) = relu_tanh(k_sharp*fs_pos(t) + k_sharp*trig_pos(t))
fs_neg(t+1) = relu_tanh(k_sharp*fs_neg(t) + k_sharp*trig_neg(t))
```

The gated front-block channels use the two center proximity taps:

```text
front_block_pos(t+1) = relu_tanh(C1(t) + C2(t) - (front_thr + gate_c) + gate_c*fs_pos(t) - gate_c*fs_neg(t))
front_block_neg(t+1) = relu_tanh(C1(t) + C2(t) - (front_thr + gate_c) - gate_c*fs_pos(t) + gate_c*fs_neg(t))
```

with `front_thr = 1.4`.

## 5. Nonzero Readout Weights

Only these hidden slots contribute directly to `Wout`:

```text
Wout[hit_feat]        = hit_turn
Wout[prox_left]       = heading_gain
Wout[prox_right]      = -heading_gain
Wout[safe_left]       = safety_gain_left
Wout[safe_right]      = safety_gain_right
Wout[front_block_pos] = +front_gain_mag
Wout[front_block_neg] = -front_gain_mag
Wout[shortcut_steer]  = +1
Wout[init_impulse]    = +1
```

The controller also copies this same readout row into the recurrent weights for
`dtheta`, so `dtheta(t+1)` is the clipped version of the previous step's
steering command.

## 6. Verification

Smoke-test commands:

```powershell
python braincraft\env2_player_reflex_bio2.py
python braincraft\_debug_bio2.py --steps 120 --stride 20
python braincraft\_debug_bio2_detail.py --steps 120
```

Accepted env2 score on 2026-04-17: `14.71 +/- 0.00`.

Streamline re-verification on 2026-04-20:

- `python braincraft\env2_player_reflex_bio2.py` -> `14.71 +/- 0.00`
- `python braincraft\_debug_bio2.py --steps 120 --stride 20` completed
- `python braincraft\_debug_bio2_detail.py --steps 120` completed

Reward-circuit `K` absorption re-verification on 2026-04-20:

- `python braincraft\env2_player_reflex_bio2.py` -> `14.71 +/- 0.00`

`cos_n` / `sin_n` rename (with sign flip) re-verification on 2026-04-20:

- `python braincraft\env2_player_reflex_bio2.py` -> `14.71 +/- 0.00`

Multi-step seeding window (approach A, `seed_window_k = 6`) on 2026-04-21:

- Previously-failing outer seed `101`: `14.589 / 0.363` -> `14.741 / 0.110`
- Previously-failing inner seed `311895`: `13.50, 21 hits` -> `14.71, 0 hits`
- 20-seed broader sweep (rng seed `20260421`): mean-of-means
  `14.7039 -> 14.7275`, max std `0.366 -> 0.107`, failure-tail seeds
  `1/20 -> 0/20`.
