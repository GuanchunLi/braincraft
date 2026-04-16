# Reflex Player v3 -- Environment 2

## Overview

Reflex Player v3 is a hand-crafted recurrent neural network (RNN) controller for a navigating bot in Environment 2 of the Braincraft challenge. It extends Reflex Player v2 with a single behavioral change: the **front-block steering gain** is no longer a fixed negative value but is **latched from early color evidence** gathered during the first few time steps. This allows the bot to adaptively choose its turning direction when a frontal obstacle is detected, based on which side the blue wall appears on.

The bot performs clockwise wall-following around a ring-shaped arena, with a corridor shortcut that it learns to exploit after encountering its first reward. An initial heading correction phase compensates for random starting jitter.

---

## 1. Network Architecture

The model is a discrete-time recurrent neural network with the following components:

| Component | Symbol | Dimensions |
|-----------|--------|------------|
| Input weight matrix | `W_in` | 1000 × 131 |
| Recurrent weight matrix | `W` | 1000 × 1000 |
| Output weight matrix | `W_out` | 1 × 1000 |
| Hidden state | `X` | 1000 × 1 |
| Input vector | `I` | 131 × 1 |
| Output | `O` | scalar |

### 1.1 Input Vector

The input `I` has `2p + 3 = 131` dimensions (where `p = 64` is the camera resolution):

```
I = [ 1 − d_0, ..., 1 − d_63,  c_0, ..., c_63,  hit,  energy,  1.0 ]ᵀ
```

- `d_i ∈ [0, 1]`: raw depth from camera ray `i` (inverted so closer = higher)
- `c_i`: color/material ID from camera ray `i`
- `hit`: 1 if the bot collided on the previous step, 0 otherwise
- `energy`: current energy level
- `1.0`: constant bias term

### 1.2 State Update Rule

At each time step `t`, the hidden state is updated as:

```
X⁽ᵗ⁾ = (1 − λ) · X⁽ᵗ⁻¹⁾  +  λ · f( W_in · I⁽ᵗ⁾  +  W · X⁽ᵗ⁻¹⁾ )
```

where `λ = 1.0` (leak rate), so the update simplifies to:

```
X⁽ᵗ⁾ = f( W_in · I⁽ᵗ⁾  +  W · X⁽ᵗ⁻¹⁾ )
```

The output is:

```
O⁽ᵗ⁾ = W_out · g( X⁽ᵗ⁾ )
```

where `g` is the identity function. The scalar output `O` is a steering angle increment (`Δθ`).

---

## 2. Activation Function `f`

The activation function `f` is a **custom, per-neuron** function that applies different logic to different hidden units. Let `z = W_in · I + W · X⁽ᵗ⁻¹⁾` be the pre-activation vector.

### 2.1 Default Activation: `relu_tanh`

Most neurons use a rectified hyperbolic tangent:

```
relu_tanh(z) = max( 0,  tanh(z) )
```

### 2.2 Neuron Roles

The 1000 hidden units are partitioned by function. Only a small fraction are actively wired; the rest remain zero throughout.

| Neuron(s) | Name | Activation | Role |
|-----------|------|------------|------|
| `X_0` | Hit | relu_tanh | Collision signal |
| `X_1` | Prox-Left | relu_tanh | Left-forward proximity |
| `X_2` | Prox-Right | relu_tanh | Right-forward proximity |
| `X_3` | Safety-Left | relu_tanh | Left-side wall distance |
| `X_4` | Safety-Right | relu_tanh | Right-side wall distance |
| `X_5` | Front-Block | relu_tanh | Frontal obstacle detector |
| `X_6` | `Δθ` | custom (clamped) | Current steering output |
| `X_7` | Direction accumulator | identity | Cumulative heading `Σ Δθ` |
| `X_8, X_9` | cos/sin | custom | `cos(θ), sin(θ)` of current heading |
| `X_10, X_11` | Position | custom (gated) | Estimated `(x, y)` displacement |
| `X_12` | Hit relay | relu_tanh | Hit signal for position gating |
| `X_13` | Correction hold | custom | Latched heading correction |
| `X_14` | Correction instantaneous | identity | Current sensor asymmetry |
| `X_15` -- `X_18` | Reward circuit | relu_tanh | Energy-pulse reward detector/latch |
| `X_19` | Step counter | custom | Steps since last corridor crossing |
| `X_20` | Steering override | custom | Shortcut/correction override |
| `X_22` | Shortcut countdown | custom | Countdown for shortcut maneuver |
| `X_23` | Init correction remainder | custom | Remaining initial heading fix |
| `X_24` | Seeded flag | custom | One-shot flag for init correction |
| `X_25` -- `X_(25+p−1)` | Color copies | identity | Raw color channel copies |
| `X_left_blue` | Left-blue indicator | custom | Blue detected on left side |
| `X_right_blue` | Right-blue indicator | custom | Blue detected on right side |
| `X_evidence` | Evidence accumulator | custom | Accumulated left/right blue evidence |
| `X_front_sign` | Front-sign latch | custom | Latched sign for front gain |
| `X_signed_front` | Signed front value | custom | `sign × X_5` |

**Index computation for v3-specific neurons** (where `p = 64`):

- Color copies: `X_25` to `X_(25 + p − 1) = X_88`
- Left-blue indicator: `X_89`
- Right-blue indicator: `X_90`
- Evidence accumulator: `X_91`
- Front-sign latch: `X_92`
- Signed-front value: `X_93`

---

## 3. Feature Neurons (`X_0` -- `X_5`)

### 3.1 Input Wiring

Each feature neuron `X_i` receives a weighted combination of specific input channels:

```
z_0 = I_hit
z_1 = I_20                        (depth ray at index 20, left-forward)
z_2 = I_43                        (depth ray at index 43, right-forward)
z_3 = −I_11 + 0.75                (left-side safety, offset by target distance)
z_4 = −I_52 + 0.75                (right-side safety, offset by target distance)
z_5 = I_31 + I_32 − 1.4           (front-block: sum of two central rays minus threshold)
```

Each is passed through `relu_tanh`:

```
X_i⁽ᵗ⁾ = relu_tanh( z_i ),    i = 0, ..., 5
```

### 3.2 Output Contribution

The wall-following steering signal from the six feature neurons is:

```
O_features = Σᵢ₌₀..₅  w_i · X_i
```

with weights:

| `i` | Gain Name | Value |
|-----|-----------|-------|
| 0 | hit turn | `−10° / tanh(1)` |
| 1 | heading (left) | `−40°` |
| 2 | heading (right) | `+40°` |
| 3 | safety left | `−20°` |
| 4 | safety right | `+20°` |
| 5 | front block | `0°` (zeroed in v3; replaced by signed front) |

All angles are in radians. In v3, the `w_5 = 0` entry is replaced by the signed front-block mechanism described in Section 4.

---

## 4. Front-Block Sign Latch (v3 Extension)

This is the key difference from Reflex Player v2. Instead of a fixed negative front-block gain, v3 determines the sign from early observations of the blue wall.

### 4.1 Color Evidence Gathering

At each activation, the raw color values `c_0, ..., c_(p−1)` are copied into `X_25, ..., X_88` via `W_in`. The activation then checks for the presence of the blue wall (color value `= 4.0`) on each side:

```
left_blue  =  ∃ j ∈ {0,  ..., 31} :  c_j ≈ 4.0
right_blue =  ∃ j ∈ {32, ..., 63} :  c_j ≈ 4.0
```

From these, binary evidence signals are produced:

```
L_ev = 1   if  left_blue  ∧ ¬right_blue,   else 0
R_ev = 1   if  right_blue ∧ ¬left_blue,    else 0
```

### 4.2 Evidence Accumulation and Latching

The evidence accumulator `X_evidence` integrates over time. Let `s_prev = X_front_sign⁽ᵗ⁻¹⁾` be the previous sign latch value.

**If the sign has not yet been latched** (`|s_prev| < 0.5`):

```
X_evidence⁽ᵗ⁾ = X_evidence⁽ᵗ⁻¹⁾ + R_ev − L_ev

                    ⎧ +1   if  X_evidence⁽ᵗ⁾ ≥ +2
X_front_sign⁽ᵗ⁾ =   ⎨ −1   if  X_evidence⁽ᵗ⁾ ≤ −2
                    ⎩  0   otherwise (not yet latched)
```

The threshold of `±2` ensures the sign is only committed after consistent evidence across multiple frames.

**If the sign has been latched** (`|s_prev| ≥ 0.5`):

```
X_evidence⁽ᵗ⁾   = X_evidence⁽ᵗ⁻¹⁾    (frozen)
X_front_sign⁽ᵗ⁾ = s_prev              (held)
```

### 4.3 Signed Front-Block Contribution

The signed front value is:

```
X_signed_front⁽ᵗ⁾ = X_front_sign⁽ᵗ⁾  ×  X_5⁽ᵗ⁾
```

The total feature-based steering becomes:

```
O_features  =  [ Σᵢ₌₀..₄ w_i · X_i ]  +  [ α · X_signed_front ]
                ↑                         ↑
                heading + safety          adaptive front-block
```

where `α = 20°` (in radians, `≈ 0.349`) is the front gain magnitude (`FRONT_GAIN_MAG`).

**Interpretation**: If the blue wall is on the right, the evidence accumulator goes positive, latching `+1`. The front-block then steers the bot to the right when facing a wall. If blue is on the left, the latch is `−1` and the bot steers left. Before the latch fires, the front-block term is zero (the bot relies purely on heading and safety gains).

---

## 5. Heading Estimation

### 5.1 Correction Term

The heading correction compensates for sensor-to-heading misalignment. On the very first step (`X_7 ≈ 0`):

```
correction = X_13⁽ᵗ⁻¹⁾
```

On subsequent steps:

```
correction = X_13⁽ᵗ⁻¹⁾ − X_14⁽ᵗ⁻¹⁾
```

where `X_13` is seeded from the initial sensor asymmetry via:

```
X_13  ←  X_13_prev  +  ( I_43 − I_20 ) / 0.173
```

and `X_14` tracks the instantaneous (non-recurrent) part of the same signal.

### 5.2 Lagged and Current Heading

The **lagged heading** (before the current steering update):

```
θ_lag = X_7⁽ᵗ⁻¹⁾  +  π/2  +  correction
```

The **current heading** (after applying the steering output):

```
θ_now = X_7⁽ᵗ⁻¹⁾  +  π/2  +  correction  +  Δθ⁽ᵗ⁾
```

### 5.3 Direction and Position Tracking

Direction components:

```
X_8⁽ᵗ⁾ = cos( θ_now ),     X_9⁽ᵗ⁾ = sin( θ_now )
```

Position integration (gated by collision):

```
            ⎧ X_10⁽ᵗ⁻¹⁾ + v · cos( θ_now )    if no hit
X_10⁽ᵗ⁾ =   ⎨
            ⎩ X_10⁽ᵗ⁻¹⁾                       if hit

            ⎧ X_11⁽ᵗ⁻¹⁾ + v · sin( θ_now )    if no hit
X_11⁽ᵗ⁾ =   ⎨
            ⎩ X_11⁽ᵗ⁻¹⁾                       if hit
```

where `v = 0.01` is the bot speed.

---

## 6. Initial Heading Correction

The bot starts with direction `π/2 + ε`, where `ε ~ U(−5°, +5°)`. The initial correction circuit physically turns the bot back to exactly upward.

### 6.1 Seeding (First Activation, `X_24 < 0.5`)

```
X_23⁽ᵗ⁾ = clip( −correction,  −0.15,  0.15 )
X_24⁽ᵗ⁾ = 1.0
```

No steering override is applied (output from this step is discarded by warmup).

### 6.2 Draining (Subsequent Steps, `X_24 ≥ 0.5`)

While `|X_23| > 10⁻³`:

```
Δθ_init  = clip( X_23,  −a,  a ),    a = 5°
X_23⁽ᵗ⁾  = X_23⁽ᵗ⁻¹⁾ − Δθ_init
```

The steering override is set to cancel wall-following and apply only the correction:

```
X_20 = Δθ_init − O_features
```

---

## 7. Reward Detection Circuit (`X_15` -- `X_18`)

This circuit detects when the bot has collected a reward (energy increase).

```
X_15⁽ᵗ⁾ = relu_tanh(    0.005 · I_energy )

X_16⁽ᵗ⁾ = relu_tanh( 1000 · X_15⁽ᵗ⁻¹⁾  +  10 · X_16⁽ᵗ⁻¹⁾ )

X_17⁽ᵗ⁾ = relu_tanh(  500 · I_energy
                    − 100000 · X_15⁽ᵗ⁻¹⁾
                    +   1000 · X_16⁽ᵗ⁻¹⁾
                    −   1000.2 )

X_18⁽ᵗ⁾ = relu_tanh(   10 · X_17⁽ᵗ⁻¹⁾  +  10 · X_18⁽ᵗ⁻¹⁾ )
```

`X_18` acts as a permanent latch: once a reward pulse is detected (`X_17` fires briefly when energy increases), `X_18` saturates to 1 and stays there due to its self-recurrence. The condition `X_18 > 0.5` is used as a "has been rewarded" flag.

---

## 8. Shortcut Circuit

After the first reward is detected, the bot attempts to take a corridor shortcut on subsequent laps instead of going around the full ring.

### 8.1 Step Counter (`X_19`)

```
            ⎧ 0                  if  | X_10⁽ᵗ⁻¹⁾ | < 0.05  AND  | sin( θ_lag ) | > 0.70
X_19⁽ᵗ⁾ =   ⎨
            ⎩ X_19⁽ᵗ⁻¹⁾ + 1      otherwise
```

The counter resets when the bot is near the center (`x ≈ 0`) and heading vertically (i.e., traversing the corridor).

### 8.2 Shortcut Trigger

The shortcut fires when **all** of these conditions hold simultaneously:

1. `X_18 > 0.5` (reward has been seen)
2. `| sin( θ_lag ) | < 0.35` (heading is approximately horizontal)
3. `X_5 < 0.1` (no frontal obstacle)
4. `X_19 > 60` (enough steps since last corridor)
5. `| X_10 − δ | < 0.05`, where `δ = −0.115 · sign( cos θ_lag )` (near corridor entrance, arc-corrected)
6. `X_22 < 0.5` (not already in a shortcut)

### 8.3 Shortcut Execution (`X_22` Countdown)

When triggered, `X_22` is set to `63` (`= 18 turn steps + 45 approach steps`) and decrements by 1 each step.

**Turning phase** (`X_22 > 45.5`): The bot executes a sharp turn into the corridor:

```
X_20 = 2.0 × d_turn
```

where `d_turn = −sign( cos θ_lag ) · sign( X_11 )` determines the turn direction (the value saturates at the `±5°` clamp).

**Approach phase** (`0.5 < X_22 ≤ 45.5`): The bot drives straight, canceling all feature contributions except the front-block:

```
X_20 = −O_no_front,    O_no_front = Σᵢ₌₀..₄  w_i · X_i
```

---

## 9. Final Output Computation

The final steering angle per time step is computed as:

```
O_now  = O_features + X_20
Δθ⁽ᵗ⁾  = clip( O_now,  −5°,  +5° )
```

This clamped value is stored in `X_6` and fed back through the recurrence into `X_7` (direction accumulator) at the next step:

```
X_7⁽ᵗ⁾ = X_7⁽ᵗ⁻¹⁾ + X_6⁽ᵗ⁾
```

The output scalar `O = W_out · X` drives the bot's `forward()` method, which applies the steering angle and advances the bot by one step.

---

## 10. Summary of v3 vs v2

| Aspect | Reflex v2 | Reflex v3 |
|--------|-----------|-----------|
| Front-block sign | Fixed `−20°` | Latched from blue-wall evidence |
| Color channels | Unused | Read during early steps |
| Additional neurons | -- | 5 extra (color copies, evidence, sign latch, signed output) |
| Behavior | Always turns same direction at walls | Adapts turn direction to arena layout |

The front-sign latch makes v3 robust to arena configurations where the optimal turn direction depends on which side the blue wall is on -- information that v2 ignores entirely.
