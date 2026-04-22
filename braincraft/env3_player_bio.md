# Bio Player — Environment 3

## 1. Overview

`env3_player_bio.py` is a pointwise-activation Echo State Network controller
for Environment 3 of the BrainCraft challenge. Every hidden activation is a
scalar function of its own preactivation; all cross-neuron logic is expressed
by the connectivity matrices. The network runs

```text
X(t+1) = f(Win @ I(t) + W @ X(t))
O(t+1) = Wout @ g(X(t+1))
```

with identity readout `g(x) = x`. The model is produced by a single `yield`
in `bio_player()` (no iterative training), so the matrices are fixed at
build time.

The controller keeps two behaviours from the env1 bio player and drops the
other two:

- **Reflex wall-follower.** Proximity, safety, and hit channels feed a
  small fixed steering readout that tracks the outer corridor.
- **Initial-heading correction.** A six-step seed latch closes a
  proportional loop that erases the `±5°` start-direction perturbation.

Dropped circuits (relative to `env1_player_bio.py`):

- **Reward-latch circuit** (`energy_ramp`, `reward_pulse`, `reward_latch`):
  no longer needed because there is no shortcut to arm.
- **Shortcut circuit** (countdown, phase gates, corridor predicates,
  quadrant ANDs, heading/position integrators, trig helpers): the env3
  agent simply runs around the outer circuit of the arena, so none of
  the pose-triggered shortcut machinery is used.

Every neuron that existed only to support those circuits is removed, so
the hidden pool shrinks from 41 wired slots in env1 to **12 wired
slots** here. Slots `12..999` receive no incoming weights.

Env3 exposes colour input, but this controller does not read it. The
full `2p + 3` env3 input layout is kept so the challenge runner feeds
the network correctly; only proximity, hit, and bias columns receive
any input weight.

## 2. Network shape

| Parameter    | Value                          |
| ------------ | ------------------------------ |
| `n`          | `1000`                         |
| `p`          | `64` (camera rays)             |
| `n_inputs`   | `2 * p + 3 = 131`              |
| `warmup`     | `0`                            |
| `leak` (λ)   | `1.0`                          |
| `g`          | identity                       |
| Actuator clip | `step_a = 5°`                 |
| Bot speed    | `0.01` (from `bot.speed`)      |

## 3. Activation library

Each neuron `i` has a fixed scalar activation applied pointwise to its
preactivation `z_i(t)`:

| Name        | Formula                       | Used by                       |
| ----------- | ----------------------------- | ----------------------------- |
| `relu_tanh` | `max(0, tanh(z))`             | default — reflex channels, seed gates, latch, front-block detector |
| `identity`  | `z`                           | `step_counter`, `init_impulse` |
| `clip_a`    | `clip(z, -step_a, +step_a)`   | `dtheta`                       |

No other activation primitives (`sin`, `square`, `bump`, `relu`) are
needed once the shortcut/reward circuitry is removed.

Module constants:

```text
front_gain_mag = 20°
k_sharp        = 50.0
step_a         = 5°
seed_window_k  = 6
cal_gain       = 1 / 0.173
front_thr      = 1.4
safety_target  = 0.75
hit_turn       = -10° / tanh(1)
heading_gain   = -40°
safety_gain_left  = -20°
safety_gain_right = +20°
```

## 4. Inputs and slot layout

The env3 challenge feeds

```text
I(t) = [prox[0..63](t), colour[0..63](t), hit(t), energy(t), 1]
```

The controller uses only the proximity block, the hit flag, and the
bias constant; colour (`p..2p-1`) and energy (`2p + 1`) are ignored.

Index aliases:

```text
L_idx          = 20        (left reflex proximity tap)
R_idx          = 43        (right reflex proximity tap)
left_side_idx  = 11        (left safety tap)
right_side_idx = 52        (right safety tap)
C1_idx, C2_idx = 31, 32    (two centre-front proximity taps)
hit_idx        = 128       (= 2p)
energy_idx     = 129       (= 2p + 1, unused)
bias_idx       = 130       (= 2p + 2)
```

Hidden slots in allocator order (`n = 1000`; slots `12..999` receive
no incoming weights):

| Slot | Name            | Activation  | Role                                                     |
| ---- | --------------- | ----------- | -------------------------------------------------------- |
| 0    | `hit_feat`      | `relu_tanh` | hit reflex                                               |
| 1    | `prox_left`     | `relu_tanh` | left proximity reflex                                    |
| 2    | `prox_right`    | `relu_tanh` | right proximity reflex                                   |
| 3    | `safe_left`     | `relu_tanh` | left safety feature                                      |
| 4    | `safe_right`    | `relu_tanh` | right safety feature                                     |
| 5    | `dtheta`        | `clip_a`    | clipped one-step-lagged steering command                 |
| 6    | `step_counter`  | `identity`  | step index (drives `seeded_flag` timing)                 |
| 7    | `seeded_flag`   | `relu_tanh` | one-way seed latch                                       |
| 8    | `seed_pos`      | `relu_tanh` | positive initial-correction pulse                        |
| 9    | `seed_neg`      | `relu_tanh` | negative initial-correction pulse                        |
| 10   | `init_impulse`  | `identity`  | initial-correction steering actuator                     |
| 11   | `front_block`   | `relu_tanh` | unsigned centre-proximity escape detector                |

## 5. Main circuits

### 5.1 Reflex features and readout

The five reflex channels are feed-forward proximity/hit detectors. Env3
never enters a shortcut-approach phase, so no silencing gate is wired:

```text
hit_feat(t+1)   = relu_tanh(hit(t))
prox_left(t+1)  = relu_tanh(prox[L_idx](t))
prox_right(t+1) = relu_tanh(prox[R_idx](t))
safe_left(t+1)  = relu_tanh(-prox[left_side_idx](t)  + safety_target)
safe_right(t+1) = relu_tanh(-prox[right_side_idx](t) + safety_target)
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
  + init_impulse(t+1)
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

implemented by mirroring `Wout` row 0 into `W[dtheta, :]`. No other
neuron reads `dtheta` in this minimal network — it is kept as an
observable internal variable mirroring the output.

### 5.2 Initial-heading correction

The raw signed correction is the depth asymmetry

```text
current_corr(t) = (prox[R_idx](t) - prox[L_idx](t)) * cal_gain
```

with `cal_gain = 1 / 0.173`. The latch is built from `step_counter`,
`seeded_flag`, `seed_pos`, `seed_neg`, and `init_impulse`:

```text
step_counter(t+1) = step_counter(t) + 1
seeded_flag(t+1)  = relu_tanh(k_sharp * (step_counter(t) - (seed_window_k - 1.5)))

seed_pos(t+1) = relu_tanh(-cal_gain*prox[L_idx](t) + cal_gain*prox[R_idx](t)
                          - 1000 * seeded_flag(t))
seed_neg(t+1) = relu_tanh( cal_gain*prox[L_idx](t) - cal_gain*prox[R_idx](t)
                          - 1000 * seeded_flag(t))

init_impulse(t+1) = -seed_pos(t) + seed_neg(t)
```

With `seed_window_k = 6`, `seeded_flag(t) = 0` for `t = 0..5` and
saturates to `1` for `t ≥ 6`. Because `seed_pos(t+1)` reads
`seeded_flag(t)`, the seeds fire for six consecutive network steps;
after that the `-1000 * seeded_flag` gate drives both seeds to zero.

`init_impulse` is read directly by the output (weight `+1`) during the
seed window, so each step adds `-(R - L) / 0.173` radians to the
steering command. Across six steps the bot rotates until the left and
right proximity taps agree, removing the random `±5°` spawn offset.

Note that env1's `head_corr` integrator is dropped here: it existed
only to correct the trig estimate of `phi` used by the shortcut's
position integrators, and no such pose estimate is computed now.

### 5.3 Front block

Env3 provides colour, but the wall-follower does not use it. A single
unsigned centre detector handles nose-on walls:

```text
front_block(t+1) = relu_tanh(C1(t) + C2(t) - front_thr)
```

with `front_thr = 1.4`. A positive reading turns the bot in a fixed
direction (`+front_gain_mag = +20°`) via the readout — a local
"bounce-off-the-wall" escape that biases the bot consistently onto
the outer corridor.

## 6. Nonzero readout weights

```text
Wout[hit_feat]        = hit_turn          = -10° / tanh(1)
Wout[prox_left]       = heading_gain      = -40°
Wout[prox_right]      = -heading_gain     = +40°
Wout[safe_left]       = safety_gain_left  = -20°
Wout[safe_right]      = safety_gain_right = +20°
Wout[front_block]     = +front_gain_mag   = +20°
Wout[init_impulse]    = +1
```

Seven nonzero weights total. The controller also mirrors this row into
`W[dtheta, :]`, so `dtheta(t+1)` is the clipped copy of the previous
step's output.

## 7. Verification

```bash
python braincraft/env3_player_bio.py
```

Runs `train(bio_player, timeout=100)` followed by
`evaluate(model, Bot, Environment, debug=False, seed=12345)` on 10
runs and prints the mean distance and standard deviation.

Expected final score: `~14.40 ± 0.49` (seed `12345`, 10 runs).

Sanity invariants for the built matrices:

- `Win.shape == (1000, 131)` and `W.shape == (1000, 1000)`.
- Only slots `0..11` receive any incoming weights.
- `Wout` has exactly seven nonzero entries (listed in §6).
