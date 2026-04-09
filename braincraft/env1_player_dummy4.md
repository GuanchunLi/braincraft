# `env1_player_dummy4.py` -- Model Structure

A deterministic, hand-wired Task-1 controller built inside the standard
1000-neuron Braincraft RNN interface.  The **steering output is identical to
[dummy2](env1_player_dummy2.py)**, but the reservoir contains an additional
internal-state subcircuit (neurons `X6`--`X14`) that maintains a running
estimate of the bot's **heading direction** and **position** using only
information available through the standard input vector -- no direct readout
from the `Bot` class.

The internal model uses three key techniques:

1. **Zero-lag direction tracking** -- the current step's $\Delta\theta$ is
   recomputed inside the activation function from the freshly-activated
   feature neurons, bypassing the 1-step delay that would otherwise arise
   from the recurrent matrix.
2. **Dead-reckoning position integration** -- displacement is accumulated
   step-by-step via $\cos/\sin$ of the tracked direction, with the
   accumulator frozen on hit steps.
3. **Sample-and-hold heading calibration** -- the unknown $\pm 5^\circ$
   initial-direction noise is estimated from the L/R sensor asymmetry at the
   known starting position and latched for the remainder of the run.

---

## 1. The generic Braincraft RNN

Task-1 players are evaluated by [challenge_1.py](challenge_1.py) with the
standard update rule:

$$
\mathbf{x}_{t+1}
=
(1-\lambda)\,\mathbf{x}_t
+
\lambda\,
f\!\left(W_{\text{in}}\mathbf{i}_t + W\mathbf{x}_t\right)
$$

$$
o_t = W_{\text{out}}\,g(\mathbf{x}_{t+1})
$$

with:

| symbol | shape | meaning |
|---|---|---|
| $\mathbf{i}_t$ | $(p+3,1)$ | input vector at step $t$ |
| $\mathbf{x}_t$ | $(n,1)$ | hidden state, here $n=1000$ |
| $W_{\text{in}}$ | $(n,p+3)$ | input weights |
| $W$ | $(n,n)$ | recurrent weights |
| $W_{\text{out}}$ | $(1,n)$ | readout |
| $\lambda$ | scalar | leak |
| $f,g$ | element-wise | hidden / readout nonlinearities |
| $o_t$ | scalar | desired heading change in radians |

For `dummy4`:

- $n = 1000$
- $p = 64$
- $\lambda = 1.0$
- `warmup = 0`
- $f$ = **custom per-neuron activation** (see Section 5)
- $g$ = `identity`

Since `leak = 1`, the update simplifies to:

$$
\mathbf{x}_{t+1}
=
f\!\left(W_{\text{in}}\mathbf{i}_t + W\mathbf{x}_t\right)
$$

The output is passed to `bot.forward(...)`, which clips the actual turn to
$\pm 5^\circ$ per step.

---

## 2. Inputs

The input vector is assembled in [challenge_1.py](challenge_1.py):

$$
\mathbf{i}_t
=
\begin{bmatrix}
1-d_{t,0} \\
\vdots \\
1-d_{t,63} \\
\text{hit}_t \\
\text{energy}_t \\
1
\end{bmatrix}
\quad\in\;\mathbb{R}^{p+3},\qquad p=64.
$$

where:

- $1 - d_{t,k}$ is proximity (higher = closer).
- Camera ray $k=0$ is the **leftmost**, $k=63$ the **rightmost**.
- Index $p=64$ is the binary collision flag, $p+1=65$ is the energy gauge,
  $p+2=66$ is a constant bias of $1$.

Define shorthand:

$$
\text{prox}_{t,k} \equiv 1 - d_{t,k}
$$

---

## 3. Sparse state layout

Only the first 15 neurons carry signal:

| neuron | role | activation | contributes to output? |
|---|---|---|---|
| `X0` | hit detector | `relu_tanh` | yes |
| `X1` | left symmetric-front ray (prox[20]) | `relu_tanh` | yes |
| `X2` | right symmetric-front ray (prox[43]) | `relu_tanh` | yes |
| `X3` | left safety (prox[11] - target) | `relu_tanh` | yes |
| `X4` | right safety (prox[52] - target) | `relu_tanh` | yes |
| `X5` | front-block detector | `relu_tanh` | yes |
| `X6` | current clamped $\Delta\theta$ | clip $\pm 5^\circ$ | no |
| `X7` | direction accumulator | identity | no |
| `X8` | $\cos(\hat\theta)$ | computed | no |
| `X9` | $\sin(\hat\theta)$ | computed | no |
| `X10` | $x$-displacement accumulator | conditional | no |
| `X11` | $y$-displacement accumulator | conditional | no |
| `X12` | hit relay | `relu_tanh` | no |
| `X13` | latched heading correction | sample-and-hold | no |
| `X14` | instantaneous sensor correction | identity | no |

All neurons `X15` through `X999` remain identically zero.

The steering output $W_{\text{out}}$ is non-zero only for `X0`--`X5`, so the
internal tracking circuit has **zero contribution to the control law**.

---

## 4. The steering law (identical to dummy2)

The readout gains are:

| term | code name | value |
|---|---|---|
| hit term | `hit_turn` | $-10^\circ / \tanh(1)$ |
| heading term | `heading_gain` | $-40^\circ$ |
| left safety | `safety_gain_left` | $-20^\circ$ |
| right safety | `safety_gain_right` | $+20^\circ$ |
| front block | `front_gain` | $-20^\circ$ |

The output is:

$$
o_t
=
\text{hit\_turn}\,x_{t+1,0}
+
\text{heading\_gain}\,(x_{t+1,1} - x_{t+1,2})
+
\text{safety\_gain\_left}\,x_{t+1,3}
+
\text{safety\_gain\_right}\,x_{t+1,4}
+
\text{front\_gain}\,x_{t+1,5}
$$

This is byte-for-byte the same as [dummy2](env1_player_dummy2.py); see that
file's documentation for a detailed behavioral analysis.

---

## 5. The custom activation function

Unlike `dummy2` (which uses a single `relu_tanh` for all neurons), `dummy4`
uses a **custom per-neuron activation** built by `make_activation(speed,
wout_feature_weights)`.  This is necessary because the internal-state neurons
require different activation semantics: identity accumulators, trigonometric
computations, conditional logic, and sample-and-hold latching.

The activation function receives the pre-activation vector

$$
\mathbf{z}_t = W_{\text{in}}\mathbf{i}_t + W\mathbf{x}_t
$$

and produces $\mathbf{x}_{t+1} = f(\mathbf{z}_t)$ with the following
per-neuron rules:

### 5.1 Feature neurons (`X0`--`X5`): `relu_tanh`

$$
x_{t+1,k} = \rho(z_{t,k}), \qquad k = 0,\ldots,5
$$

where $\rho(z) = \max(\tanh(z), 0)$.

### 5.2 Clamped dtheta (`X6`): zero-lag from features

Rather than reading the 1-step-delayed output from the $W$ matrix, `X6`
recomputes the current output $O_t$ directly from the just-activated feature
neurons:

$$
O_t^{\text{now}} = \sum_{k=0}^{5} W_{\text{out}}[0,k] \cdot x_{t+1,k}
$$

$$
x_{t+1,6} = \mathrm{clip}\!\left(O_t^{\text{now}},\;-\frac{5\pi}{180},\;
+\frac{5\pi}{180}\right)
$$

This is the clamped steering angle that the bot will actually execute, computed
with **zero lag** (same step, not delayed by one).

### 5.3 Direction accumulator (`X7`): identity with self-recurrence

From the $W$ matrix: $z_{t,7} = x_{t,7} + x_{t,6}$, so:

$$
x_{t+1,7} = z_{t,7} = x_{t,7} + x_{t,6} = \sum_{\tau=0}^{t-1} \Delta\hat\theta_\tau
$$

This is the cumulative sum of all past clamped dtheta values (steps $0$
through $t-1$), i.e. a 1-step-lagged direction accumulator.

The activation function then computes a **zero-lag** direction estimate by
adding the current step's $\Delta\hat\theta_t$:

$$
\hat\theta_t = x_{t+1,7} + \frac{\pi}{2} + c_t + \Delta\hat\theta_t
$$

where $c_t$ is the heading correction from the calibration circuit
(Section 6) and $\pi/2$ is the known initial heading (north).

### 5.4 Direction readout (`X8`, `X9`): cos/sin

$$
x_{t+1,8} = \cos(\hat\theta_t), \qquad
x_{t+1,9} = \sin(\hat\theta_t)
$$

These are computed directly in the activation function (no $W$ or $W_{\text{in}}$
entries).  They provide the zero-lag direction estimate in a form suitable for
position integration and for external readout via `atan2(X9, X8)`.

### 5.5 Position accumulators (`X10`, `X11`): hit-gated dead reckoning

From the $W$ matrix: $W[10,10] = W[11,11] = 1$ (self-recurrence only).

The activation function conditionally integrates position using the hit
signal from `X12`:

$$
x_{t+1,10} =
\begin{cases}
z_{t,10} + v\cos(\hat\theta_t) & \text{if } x_{t+1,12} < 0.5 \\
z_{t,10} & \text{otherwise (hit: freeze)}
\end{cases}
$$

$$
x_{t+1,11} =
\begin{cases}
z_{t,11} + v\sin(\hat\theta_t) & \text{if } x_{t+1,12} < 0.5 \\
z_{t,11} & \text{otherwise (hit: freeze)}
\end{cases}
$$

where $v = 0.01$ is the bot speed.

The model position is then:

$$
\hat{p}_x = 0.5 + x_{t+1,10}, \qquad
\hat{p}_y = 0.5 + x_{t+1,11}
$$

The offset $0.5$ accounts for the known starting position $(0.5, 0.5)$.

### 5.6 Hit relay (`X12`): `relu_tanh`

$$
x_{t+1,12} = \rho(z_{t,12}) = \rho(\text{hit}_t)
$$

A simple relay of the hit input, used to gate the position accumulators.

---

## 6. Heading calibration circuit (`X13`, `X14`)

### 6.1 The problem

The bot's initial direction is $\theta_0 = \pi/2 + \epsilon$, where $\epsilon
\sim \text{Uniform}(-5^\circ, +5^\circ)$ is unknown.  Without correcting for
$\epsilon$, the direction model starts with an error equal to the noise, and
since it is purely integrating dtheta, this bias persists forever and
contaminates position tracking.

### 6.2 The idea

At the known starting position $(0.5, 0.5)$ facing approximately north, a
heading offset $\epsilon$ creates a measurable asymmetry in the left/right
proximity readings.  Specifically, with the north wall at $y = 0.9$, a
positive $\epsilon$ (CCW) shifts the left ray closer to the wall and the right
ray further, producing a negative $(\text{prox}_R - \text{prox}_L)$.

The empirically measured sensitivity is:

$$
\frac{\partial(\text{prox}_R - \text{prox}_L)}{\partial\epsilon}
\;\approx\; 0.173 \;\text{(radians}^{-1}\text{)}
$$

So the correction estimate is:

$$
\hat\epsilon = \frac{\text{prox}_{0,R} - \text{prox}_{0,L}}{0.173}
$$

### 6.3 The circuit

Two neurons implement a sample-and-hold mechanism:

**X13** (latched correction) has self-recurrence and sensor input:

$$
W[13,13] = 1, \quad
W_{\text{in}}[13, 20] = -\text{cal}, \quad
W_{\text{in}}[13, 43] = +\text{cal}
$$

where $\text{cal} = 1/0.173 \approx 5.78$.

**X14** (instantaneous) has the same $W_{\text{in}}$ weights but **no
self-recurrence**:

$$
W_{\text{in}}[14, 20] = -\text{cal}, \quad
W_{\text{in}}[14, 43] = +\text{cal}
$$

The pre-activations are therefore:

$$
z_{t,13} = x_{t,13} + \text{cal}\cdot(\text{prox}_{t,43} - \text{prox}_{t,20})
$$

$$
z_{t,14} = \text{cal}\cdot(\text{prox}_{t,43} - \text{prox}_{t,20})
$$

### 6.4 Step-0 detection

The activation function detects step 0 by checking whether the direction
accumulator $x_{t+1,7}$ is zero (which it is only at $t=0$, since `X6`
stores the nonzero current dtheta from step 0 onward).

**At step 0** ($x_{t+1,7} = 0$):

$$
c_0 = z_{0,13} = \text{cal}\cdot(\text{prox}_{0,43} - \text{prox}_{0,20})
\;\approx\; \hat\epsilon
$$

The activation function latches this value: $x_{1,13} \leftarrow c_0$.

**At step $t \ge 1$** ($x_{t+1,7} \ne 0$):

$$
c_t = z_{t,13} - z_{t,14}
= x_{t,13} + \text{cal}\cdot\Delta_t - \text{cal}\cdot\Delta_t
= x_{t,13}
$$

where $\Delta_t = \text{prox}_{t,43} - \text{prox}_{t,20}$.  The subtraction
cancels the current-step sensor contribution, recovering the latched
first-step value $c_0$ regardless of the bot's current sensor state.

The activation function preserves the latch: $x_{t+1,13} \leftarrow c_t = c_0$.

### 6.5 Why two neurons are needed

A single self-recurrent neuron would accumulate the sensor asymmetry at every
step, not just step 0.  The second neuron (`X14`) provides the instantaneous
sensor value that must be subtracted to isolate the latched component.  This
is a classic **sample-and-hold** pattern implemented within the RNN framework.

---

## 7. Timing diagram

Let $\epsilon$ be the unknown initial heading noise.  Here is what happens on
the first few steps:

### Step 0

1. Input $\mathbf{i}_0$ is read from the initial camera state (bot at
   $(0.5, 0.5)$ heading $\pi/2 + \epsilon$).
2. Features $x_{1,0}\ldots x_{1,5}$ are activated via `relu_tanh`.
3. $\Delta\hat\theta_0 = \text{clip}(W_{\text{out}} \cdot [x_{1,0:6}])$ is
   computed in the activation function.
4. $x_{1,6} = \Delta\hat\theta_0$ (stored for accumulator).
5. $x_{1,7} = 0$ (accumulator starts empty).
6. Since $x_{1,7} = 0$: calibration fires, $c_0 \approx \hat\epsilon$.
7. $\hat\theta_0 = 0 + \pi/2 + c_0 + \Delta\hat\theta_0 \approx \pi/2 +
   \epsilon + \Delta\hat\theta_0$.
8. $x_{1,8} = \cos(\hat\theta_0)$, $x_{1,9} = \sin(\hat\theta_0)$.
9. Position integrates: $x_{1,10} = v\cos(\hat\theta_0)$,
   $x_{1,11} = v\sin(\hat\theta_0)$.
10. Bot executes $\text{bot.forward}(o_0, \ldots)$, advancing to a new
    position and heading.

### Step 1

1. Input $\mathbf{i}_1$ reflects the bot's new state.
2. $x_{2,7} = x_{1,7} + x_{1,6} = 0 + \Delta\hat\theta_0 \ne 0$.
3. Since $x_{2,7} \ne 0$: calibration uses latched value
   $c_1 = z_{1,13} - z_{1,14} = c_0$.
4. $\hat\theta_1 = \Delta\hat\theta_0 + \pi/2 + c_0 + \Delta\hat\theta_1
   = \pi/2 + \epsilon + \sum_{\tau=0}^{1}\Delta\hat\theta_\tau$.
5. Position continues integrating.

### Step $t \ge 1$

The general formula for the zero-lag direction estimate is:

$$
\hat\theta_t
= \frac{\pi}{2} + c_0 + \sum_{\tau=0}^{t} \Delta\hat\theta_\tau
\approx \frac{\pi}{2} + \epsilon + \sum_{\tau=0}^{t} \Delta\hat\theta_\tau
= \theta_t^{\text{true}}
$$

The approximation $c_0 \approx \epsilon$ relies on the calibration gain being
correctly tuned to the sensor geometry at the starting position.

---

## 8. Validation results

Running `python env1_player_dummy4.py` tests the internal model against
ground truth from the `Bot` class across 4 random seeds:

| metric | typical value |
|---|---|
| Direction error (mean) | $\sim 0.06^\circ$--$0.10^\circ$ |
| Direction error (max) | $\sim 2^\circ$--$5^\circ$ (step 0 only) |
| Direction error (final) | $< 0.1^\circ$ |
| Position error (mean) | $\sim 0.010$ |
| Position error (max) | $\sim 0.010$ |
| Wall hits | 0 |

The max direction error occurs only at step 0 (before calibration takes
effect) and equals the initial noise magnitude.  From step 1 onward the
direction error stays well below $0.2^\circ$.

The position error of $\sim 0.01$ is a small constant offset that accumulates
during the first step (before calibration) and then remains approximately
stable.

### 8.1 Sources of residual error

1. **1-step calibration delay**: at step 0, the correction $c_0$ is computed
   and applied, but the direction accumulator $x_{t+1,7}$ is still zero, so
   position integration uses the uncorrected-then-corrected direction for one
   step.

2. **Calibration gain approximation**: the gain $1/0.173$ is derived from the
   sensor geometry at the exact starting position and heading.  Small
   nonlinearities in the ray-depth function mean the gain is not perfectly
   constant across the $\pm 5^\circ$ noise range.

3. **Dead-reckoning drift**: position is integrated via discrete Euler steps
   of the modeled direction.  Any direction error (even transient) causes a
   permanent position offset.

Despite these, the errors are small enough that the internal model provides
a reliable estimate of bot state throughout the entire run.

---

## 9. Relationship to other dummy players

| player | steering | internal state |
|---|---|---|
| [dummy](env1_player_dummy.py) | P-controller on left side ray | none |
| [dummy2](env1_player_dummy2.py) | heading symmetry + safety + front-block | none |
| [dummy3](env1_player_dummy3.py) | same as dummy2 (with `heading_gain=0`) | reward detection (`X6`--`X10`) |
| **dummy4** | **same as dummy2** | **direction & position tracking (`X6`--`X14`)** |

The key insight of `dummy4` is that heading and position can be reliably
tracked using only the standard RNN interface (no privileged access to
`bot.direction` or `bot.position`), by:

- accumulating the self-computed steering output (which the bot is known to
  clip and apply),
- calibrating the unknown initial heading from a single sensor reading at a
  known position,
- dead-reckoning position from the tracked heading and known speed.

---

## 10. File map

| file | role |
|---|---|
| [env1_player_dummy4.py](env1_player_dummy4.py) | controller + internal tracking circuit |
| [env1_player_dummy2.py](env1_player_dummy2.py) | steering law source (identical output) |
| [env1_trajectory_plot.py](env1_trajectory_plot.py) | trajectory visualization (includes `dummy4` in PLAYERS) |
| [env1_trajectory_variable_plot.py](env1_trajectory_variable_plot.py) | per-step variable plotting (use `--var "X[8,0]"` for cos, `--var "X[10,0]"` for model x, etc.) |
| [challenge_1.py](challenge_1.py) | RNN update rule and evaluation loop |
| [bot.py](bot.py) | $\pm 5^\circ$ clamp, kinematics, speed = 0.01 |
| [camera.py](camera.py) | 64-ray depth sensing, ray ordering |
| [environment_1.py](environment_1.py) | 10x10 world with walls and energy sources |
