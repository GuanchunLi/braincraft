# Bio Player — Environment 3

## 1. Overview

`env3_player_bio.py` is a pointwise-activation Echo State Network
controller for Environment 3. Every hidden activation is a scalar
function of its own preactivation; all cross-neuron logic lives in the
connectivity matrices:

```text
X(t+1) = f(Win @ I(t) + W @ X(t))
O(t+1) = Wout @ g(X(t+1))        (g = identity)
```

The model is produced by a single `yield` in `bio_player()`, so the
matrices are fixed at build time (no iterative training).

Env3 exposes colour (the sources are coloured), but this controller
does not read colour or energy. The bot simply runs around the outer
corridor with a reflex wall-follower and picks up whatever source it
happens to cross. Compared with the env1 bio player (`env1_player_bio.py`,
see `env1_player_bio.md`), three full circuits are removed:

- the **initial-heading correction latch** (seed_pos/neg, head_corr,
  init_impulse, step_counter, seeded_flag),
- the **rising-edge reward circuit** (energy_ramp, reward_pulse,
  reward_latch), and
- the **pose-gated corridor shortcut** (sc_countdown, phase gates,
  trig_sc, near_e/w, near_cr_*, sin_n, cos_n, sin_pos/neg,
  y_pos/neg, quadrant ANDs, shortcut_steer, dir_accum, pos_x, pos_y).

Every neuron that existed only to support those three circuits is
removed, leaving **7 wired hidden slots** (slots `7..999` receive no
incoming weights).

The full `2p + 3` env3 input layout is kept so the challenge runner
feeds the network correctly; only the proximity, hit, and bias columns
receive any input weight.

## 2. Network shape

| Parameter     | Value                         |
| ------------- | ----------------------------- |
| `n`           | `1000`                        |
| `p`           | `64` (camera rays)            |
| `n_inputs`    | `2*p + 3 = 131`               |
| `warmup`      | `0`                           |
| `leak` (λ)    | `1.0`                         |
| `g`           | identity                      |
| Actuator clip | `step_a = 5°`                 |

Module constants:

```text
front_gain_mag = 20°
step_a         = 5°
```

## 3. Activation library

Each slot has a fixed scalar activation applied pointwise to its
preactivation `z`:

| Name        | Formula                     | Used by                                                 |
| ----------- | --------------------------- | ------------------------------------------------------- |
| `relu_tanh` | `max(0, tanh(z))`           | default — reflex channels (slots 0..4) and front-block  |
| `clip_a`    | `clip(z, -step_a, +step_a)` | `dtheta`                                                |

No other activation primitives (`sin`, `bump`, `relu`, `identity`)
are needed once the shortcut, reward, and heading-correction circuits
are removed.

## 4. Inputs and slot layout

```text
I(t) = [prox[0..63](t), colour[0..63](t), hit(t), energy(t), 1]
```

Input taps used by the controller (colour and energy columns are
unread; their weight rows are all zero):

```text
L_idx          = 20      (left reflex proximity tap)
R_idx          = 43      (right reflex proximity tap)
left_side_idx  = 11      (left safety tap)
right_side_idx = 52      (right safety tap)
C1_idx, C2_idx = 31, 32  (centre-front proximity taps)
hit_idx        = 128     (= 2*p)
bias_idx       = 130     (= 2*p + 2)
```

Hidden slots (`n = 1000`; slots `7..999` receive no incoming weights):

| Slot | Name          | Activation  | Role                                  |
| ---- | ------------- | ----------- | ------------------------------------- |
| 0    | `hit_feat`    | `relu_tanh` | hit reflex                            |
| 1    | `prox_left`   | `relu_tanh` | left proximity reflex                 |
| 2    | `prox_right`  | `relu_tanh` | right proximity reflex                |
| 3    | `safe_left`   | `relu_tanh` | left safety feature                   |
| 4    | `safe_right`  | `relu_tanh` | right safety feature                  |
| 5    | `dtheta`      | `clip_a`    | one-step-lagged steering command      |
| 6    | `front_block` | `relu_tanh` | unsigned front-block detector         |

## 5. Circuits

### 5.1 Reflex features and readout

The five reflex channels are feed-forward proximity/hit detectors. Env3
never enters a shortcut-approach phase, so no silencing gate is wired:

```text
hit_feat(t+1)   = relu_tanh(hit(t))
prox_left(t+1)  = relu_tanh(prox[L_idx](t))
prox_right(t+1) = relu_tanh(prox[R_idx](t))
safe_left(t+1)  = relu_tanh(-prox[left_side_idx](t)  + 0.75)
safe_right(t+1) = relu_tanh(-prox[right_side_idx](t) + 0.75)
```

Steering readout:

```text
O(t+1) = hit_turn          * hit_feat(t+1)
       + heading_gain      * prox_left(t+1)
       - heading_gain      * prox_right(t+1)
       + safety_gain_left  * safe_left(t+1)
       + safety_gain_right * safe_right(t+1)
       + front_gain_mag    * front_block(t+1)
```

with

```text
hit_turn          = -10° / tanh(1)
heading_gain      = -40°
safety_gain_left  = -20°
safety_gain_right = +20°
front_gain_mag    = +20°
```

The `dtheta` slot holds the clipped one-step-lagged command,
`dtheta(t+1) = clip(O(t), ±step_a)`, implemented by mirroring `Wout`
row 0 into `W[dtheta, :]`. It is not read by any other neuron but is
kept so the readout is observable as a hidden-state variable.

### 5.2 Front block

An unsigned front-block from the two centre proximity taps:

```text
front_block(t+1) = relu_tanh(prox[C1_idx](t) + prox[C2_idx](t) - front_thr)
```

with `front_thr = 1.4`. A positive reading turns the bot by
`+front_gain_mag = +20°` (CCW) — a fixed-direction escape that biases
the bot consistently onto the outer corridor.

## 6. Nonzero readout weights

```text
Wout[hit_feat]    = hit_turn          = -10° / tanh(1)
Wout[prox_left]   = heading_gain      = -40°
Wout[prox_right]  = -heading_gain     = +40°
Wout[safe_left]   = safety_gain_left  = -20°
Wout[safe_right]  = safety_gain_right = +20°
Wout[front_block] = +front_gain_mag   = +20°
```

Six nonzero readout weights total. The same row is mirrored into
`W[dtheta, :]` so that `dtheta(t+1) = clip(O(t), ±step_a)`.

## 7. Verification

```bash
python braincraft/env3_player_bio.py
```

Runs `train(bio_player, timeout=100)` followed by
`evaluate(model, Bot, Environment, debug=False, seed=12345)` over 10
episodes. Observed output:

```text
Final score (distance): 14.40 +/- 0.49
```

Sanity invariants for the built matrices:

- `Win.shape == (1000, 131)` and `W.shape == (1000, 1000)`.
- Only slots `0..6` receive any incoming weights.
- `Wout` has exactly six nonzero entries (listed in §6).
