# Reflex Player v2 — Technical Description

Extends reflex player v1 with a **shortcut steering circuit** that
redirects the bot through the inner corridor after receiving the first
reward, creating a half-ring lap pattern instead of a full ring.  The
base wall-following controller and internal-state subsystems (X0-X18)
are identical to v1; this document focuses on the new shortcut circuit
(X19-X22) and describes the full system for completeness.


## Arena layout

The Environment 1 world is a 10x10 grid (each cell spans 0.1 in
world coordinates).  Two internal wall blocks (columns 3 and 6, rows
3-6) divide the interior into an outer ring and a central **corridor**
(columns 4-5) running vertically:

```
  1 1 1 1 1 1 1 1 1 1
  1 0 0 0 0 0 0 0 0 1
  1 0 0 0 0 0 0 0 0 1
  1 0 0 1 0 0 1 0 0 1       0 = open
  1 S S 1 0 0 1 T T 1       1 = wall
  1 S S 1 0 0 1 T T 1       S = source -1 (left)
  1 0 0 1 0 0 1 0 0 1       T = source -2 (right)
  1 0 0 0 0 0 0 0 0 1
  1 0 0 0 0 0 0 0 0 1
  1 1 1 1 1 1 1 1 1 1
```

The corridor (columns 4-5, rows 3-6) is 0.2 units wide and 0.4 units
tall.  The bot (starting at the center `(0.5, 0.5)`) can traverse it
to switch between the left and right halves of the ring without
completing a full lap.


## Framework

Identical to reflex v1.  At each step:

```
X(t+1) = f( W_in I(t) + W X(t) )        leak = 1
O(t)   = W_out X(t)                      g = identity
```

- `I(t)`: 67-element input (64 camera rays + hit + energy + bias=1)
- `X(t)`: 1000-element state vector
- `O(t)`: scalar steering command (radians), clipped to +/-5 deg
- `f`: per-neuron activation (see neuron map)


## Input vector `I(t)`

| Index   | Signal                                                         |
|---------|----------------------------------------------------------------|
| 0 - 63  | Camera proximity: `1 - depth` (64 rays, 0=leftmost, 63=right) |
| 64      | Hit (1 on collision, 0 otherwise)                              |
| 65      | Energy (depletes at 1/1000 per move, 5/1000 on hit)            |
| 66      | Bias (always 1.0)                                              |


## Neuron map

### Feature neurons X0-X5 (relu_tanh, drive output)

Identical to v1.  All six use `relu_tanh(x) = max(0, tanh(x))`.

| Neuron | W_in weights | Role | W_out weight |
|--------|-------------|------|-------------|
| X0 | `W_in[0, 64] = 1` | Hit signal | `w_0 = -\frac{\pi/18}{\tanh(1)} \approx -0.229` |
| X1 | `W_in[1, 20] = 1` | Left-front proximity (ray 20) | `w_1 = -2\pi/9 \approx -0.698` |
| X2 | `W_in[2, 43] = 1` | Right-front proximity (ray 43) | `w_2 = +2\pi/9 \approx +0.698` |
| X3 | `W_in[3, 11] = -1`, `W_in[3, 66] = 0.75` | Left safety | `w_3 = -\pi/9 \approx -0.349` |
| X4 | `W_in[4, 52] = -1`, `W_in[4, 66] = 0.75` | Right safety | `w_4 = +\pi/9 \approx +0.349` |
| X5 | `W_in[5, 31] = 1`, `W_in[5, 32] = 1`, `W_in[5, 66] = -1.4` | Front-block | `w_5 = -\pi/9 \approx -0.349` |

#### Wall-following output

```
O_features = sum_{i=0}^{5} w_i * relu_tanh(x_i)
```

Expanded:

```
O_features = w_0 * X0  +  w_1 * X1  +  w_2 * X2
           + w_3 * X3  +  w_4 * X4  +  w_5 * X5
```

The partial sum excluding the front-block neuron is:

```
O_no_front = sum_{i=0}^{4} w_i * relu_tanh(x_i)
```


### Head-direction & position tracking X6-X14 (custom activation, zero W_out)

Identical to v1.  These neurons internally estimate heading and
position via dead reckoning.

| Neuron | Type | W_in / W | Description |
|--------|------|----------|-------------|
| X6 | Computed | -- | Current clamped steering: `clip(O_now, -a, a)` where `a = pi/36` |
| X7 | Identity | `W[7,7]=1, W[7,6]=1` | Direction accumulator: `X7(t+1) = X7(t) + X6(t)` |
| X8 | Computed | -- | `cos(theta_now)` |
| X9 | Computed | -- | `sin(theta_now)` |
| X10 | Gated integrator | `W[10,10]=1` | x-displacement (frozen on hit) |
| X11 | Gated integrator | `W[11,11]=1` | y-displacement (frozen on hit) |
| X12 | relu_tanh | `W_in[12,64]=1` | Hit signal copy (gates X10/X11) |
| X13 | Sample-and-hold | `W[13,13]=1`, `W_in[13,20]=-c`, `W_in[13,43]=+c` | Latched heading correction |
| X14 | Identity | `W_in[14,20]=-c`, `W_in[14,43]=+c` | Instantaneous sensor asymmetry |

where `c = 1/0.173 ~ 5.78` (calibration gain).

#### Direction estimation

```
val7       = X7(t)                              (lagged cumulative dtheta)
correction = X13(t) - X14(t)                    (latched step-0 asymmetry; at step 0: X13)
theta_lag  = val7 + pi/2 + correction
theta_now  = val7 + pi/2 + correction + dtheta_now
```

where `dtheta_now = clip(O_now, -pi/36, pi/36)`.

#### Position estimation

```
if hit(t) < 0.5:
    X10(t+1) = X10(t) + v * cos(theta_now)
    X11(t+1) = X11(t) + v * sin(theta_now)
else:
    X10(t+1) = X10(t)
    X11(t+1) = X11(t)
```

where `v = 0.01` (bot speed).  World position: `(0.5 + X10, 0.5 + X11)`.


### Reward-state circuit X15-X18 (relu_tanh, zero W_out)

Identical to v1.  Detects energy refills and latches `is_rewarded`.

| Neuron | W_in / W | Description |
|--------|----------|-------------|
| X15 | `W_in[15,65] = K` | Delayed scaled energy copy |
| X16 | `W[16,15]=1000, W[16,16]=10` | Armed latch |
| X17 | `W_in[17,65]=500, W[17,15]=-100000, W[17,16]=1000, W_in[17,66]=-1000.2` | Reward-rise pulse |
| X18 | `W[18,17]=10, W[18,18]=10` | Latched is_rewarded flag |

where `K = 0.005`.  Once X18 > 0.5, it stays pinned near 1.0
indefinitely.


### NEW: Shortcut steering circuit X19-X22

These four neurons implement the corridor shortcut.  All use
self-recurrence (`W[i,i] = 1`) but their values are overwritten by
custom logic in the activation function.

#### X19 — Step counter (cooldown)

Prevents the shortcut from re-triggering immediately after exiting
the corridor.

**Update rule:**

```
if |X10(t)| < delta_center  AND  |sin(theta_lag)| > tau_vert:
    X19(t+1) = 0                                 (reset: in corridor, heading vertical)
else:
    X19(t+1) = X19(t) + 1                        (increment)
```

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Center threshold | `delta_center` | 0.05 |
| Vertical heading threshold | `tau_vert` | 0.70 |

The counter resets when the bot is near `x = 0.5` (corridor center)
and heading roughly north or south.  This places the reset zone inside
the corridor itself, ensuring the bot must travel through and away
before the shortcut can trigger again.


#### X22 — Phase countdown timer

Controls the two-phase shortcut manoeuvre.  Not connected to W_out.

**Update rule:**

```
if TRIGGER  AND  X22(t) < 0.5:
    X22(t+1) = N_total                           (= N_turn + N_approach = 63)
elif X22(t) > 0.5:
    X22(t+1) = X22(t) - 1
else:
    X22(t+1) = 0
```

Phase classification from the countdown value:

```
is_turning     = X22(t+1) > N_approach + 0.5      (i.e. countdown in [46, 63])
is_approaching = X22(t+1) > 0.5  AND  NOT is_turning   (i.e. countdown in [1, 45])
```

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Turn steps | `N_turn` | 18 |
| Approach steps | `N_approach` | 45 |
| Total | `N_total` | 63 |


#### Trigger conditions

The shortcut fires when **all five** conditions hold simultaneously:

```
TRIGGER = (a) AND (b) AND (c) AND (d) AND (e) AND (f)
```

| Cond. | Expression | Meaning |
|-------|------------|---------|
| (a) | `X18(t) > 0.5` | Reward has been collected (is_rewarded) |
| (b) | `|sin(theta_lag)| < tau_horiz` | Heading is roughly horizontal |
| (c) | `X5(t) < 0.1` | No wall directly ahead (front clear) |
| (d) | `X19(t) > N_cooldown` | Enough steps since last corridor pass |
| (e) | `|X10(t) - delta_drift| < delta_center` | x-position near corridor center (with drift compensation) |
| (f) | `X22(t) < 0.5` | No shortcut already in progress |

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Horizontal heading threshold | `tau_horiz` | 0.35 |
| Cooldown threshold | `N_cooldown` | 60 |
| Center threshold | `delta_center` | 0.05 |

**Drift offset** (`delta_drift`):

The bot follows a curved arc when wall-following, so its x-position
at the corridor entrance is slightly offset from x = 0.5 depending on
travel direction.  The offset compensates for this:

```
cos_lag = cos(theta_lag)

delta_drift = -D * sign(cos_lag)    if |cos_lag| > 0.5
              0                     otherwise
```

where `D = 0.115` (empirically calibrated as `speed / max_turn_rate`).

When heading right (`cos_lag > 0`), the trigger fires slightly to the
left of center (`X10 ~ -0.115`), and vice versa.  This ensures the
subsequent 90-degree turn arc ends with the bot aligned to the
corridor.


#### Dynamic turn direction

The turn direction is computed to always steer the bot **into** the
corridor, regardless of which side of the ring it approaches from:

```
if |X11(t)| > 0.01:
    turn_toward = -sign(cos_lag) * sign(X11(t))
else:
    turn_toward = -1.0                           (default: right turn)
```

The sign logic:
- `sign(cos_lag)`: +1 heading right, -1 heading left
- `sign(X11)`: +1 if above center (y > 0.5), -1 if below
- The product `-sign(cos) * sign(Y)` yields: heading right above
  center -> turn right (toward corridor below); heading left below
  center -> turn right (toward corridor above); etc.


#### X20 — Steering override

The core output neuron of the shortcut circuit.  Connected via
`W_out[0, 20] = 1.0`, so its value **directly adds to the steering
command**.

**Update rule (three modes):**

```
if is_turning:
    X20(t+1) = |S| * turn_toward                 (= +/-2.0)
elif is_approaching:
    X20(t+1) = -O_no_front
else:
    X20(t+1) = 0
```

where `S = -2.0` (shortcut turn magnitude).

**Total output during each phase:**

```
O(t) = clip( O_features + X20(t),  -a,  a )     where a = pi/36
```

| Phase | X20 value | Effective output |
|-------|-----------|------------------|
| Normal | `0` | `clip(O_features, -a, a)` — standard wall-following |
| Turn | `+/-2.0` | `+/-a` (saturates clamp, maximum turn rate) |
| Approach | `-O_no_front` | `clip(w_5 * X5, -a, a)` — only front-block remains active |

**Turn phase** (steps 63 down to 46):
  The large magnitude `|2.0| >> a` saturates the steering clamp to the
  maximum turn rate of `+/-pi/36` rad/step.  Over 18 steps this
  produces a cumulative turn of approximately:

```
Delta_theta = N_turn * a = 18 * pi/36 = pi/2  (90 degrees)
```

**Approach phase** (steps 45 down to 1):
  By setting `X20 = -O_no_front`, the output becomes:

```
O = O_features - O_no_front = w_5 * X5
```

  This cancels the heading, safety, and hit components, leaving only
  the front-block neuron active.  The bot drives in a straight line
  through the corridor while still being able to brake if it encounters
  a wall ahead.


#### X21 — Corridor-active countdown

Enables lateral centering correction during and after corridor
traversal.

**Update rule:**

```
if X22(t+1) > 0.5:
    X21(t+1) = N_corridor                        (= 120, reset to max)
elif X21(t) > 0.5:
    X21(t+1) = X21(t) - 1                        (count down)
else:
    X21(t+1) = 0
```

X21 stays active for up to 120 steps after the shortcut fires (longer
than the 63-step shortcut manoeuvre itself), providing centering
support during the transition back to wall-following.

**Corridor centering correction:**

When X21 is active and the bot is heading with a significant vertical
component, a lateral correction is applied:

```
if X21(t+1) > 0.5  AND  |sin(theta_lag)| > tau_corr:
    centering = G * sin(theta_lag) * (x3_pre - x4_pre)
else:
    centering = 0
```

where:
- `G = 5.0` (corridor centering gain)
- `x3_pre`, `x4_pre` are the pre-activation values of the left and
  right safety neurons (proportional to side wall proximity)
- `tau_corr = 0.30` (minimum vertical heading component)

The centering term uses `sin(theta_lag)` as a sign selector:
- Heading upward (`sin > 0`): if left wall is closer
  (`x3_pre > x4_pre`), steer right (positive correction)
- Heading downward (`sin < 0`): sign flips, maintaining the same
  wall-avoidance semantics

Note: in the current implementation, `centering` is computed but not
explicitly added to the output — the corridor centering effect is
achieved implicitly through the approach-phase cancellation leaving
only X5 active, combined with the natural sensor-driven corrections
when the bot re-enters wall-following mode.


## Shortcut execution timeline

```
Step 0 (trigger):
  X22 = 63,  X21 = 120
  is_turning = True
  X20 = +/-2.0
  Bot begins hard 90-deg turn into corridor

Steps 1-17 (turn phase continues):
  X22 counts down: 62, 61, ..., 46
  X20 = +/-2.0 (sustained max turn)
  Cumulative turn ~ 18 * 5 deg = 90 deg

Step 18 (transition to approach):
  X22 = 45
  is_turning = False, is_approaching = True
  X20 = -O_no_front
  Bot drives straight, only front-block active

Steps 19-62 (approach phase):
  X22 counts down: 44, 43, ..., 1
  Bot traverses corridor in straight line

Step 63 (shortcut complete):
  X22 = 0
  X20 = 0
  Normal wall-following resumes

Steps 64-120+ (corridor-active tail):
  X21 continues counting down
  Centering correction available during transition
```


## Complete output equation

```
O_features   = sum_{i=0}^{5} w_i * relu_tanh(x_i)

O_no_front   = sum_{i=0}^{4} w_i * relu_tanh(x_i)

                 |S| * turn_toward     if X22 > N_approach + 0.5
X20        =     -O_no_front           if 0.5 < X22 <= N_approach + 0.5
                 0                     otherwise

O          = clip( O_features + X20,  -pi/36,  pi/36 )
```


## Weight matrices summary

### W_in (n x 67) — non-zero entries

Identical to v1 (no new input connections for the shortcut circuit).

| Row | Col | Value | Purpose |
|-----|-----|-------|---------|
| 0 | 64 | 1.0 | X0: hit |
| 1 | 20 | 1.0 | X1: left-front ray |
| 2 | 43 | 1.0 | X2: right-front ray |
| 3 | 11 | -1.0 | X3: left safety (negated) |
| 3 | 66 | 0.75 | X3: safety target bias |
| 4 | 52 | -1.0 | X4: right safety (negated) |
| 4 | 66 | 0.75 | X4: safety target bias |
| 5 | 31 | 1.0 | X5: centre ray 1 |
| 5 | 32 | 1.0 | X5: centre ray 2 |
| 5 | 66 | -1.4 | X5: front threshold bias |
| 12 | 64 | 1.0 | X12: hit (position gating) |
| 13 | 20 | -5.78 | X13: -c * prox_L |
| 13 | 43 | +5.78 | X13: +c * prox_R |
| 14 | 20 | -5.78 | X14: -c * prox_L |
| 14 | 43 | +5.78 | X14: +c * prox_R |
| 15 | 65 | 0.005 | X15: scaled energy |
| 17 | 65 | 500.0 | X17: amplified energy |
| 17 | 66 | -1000.2 | X17: arm gate + threshold |

### W (n x n) — non-zero entries

New entries for v2 marked with *.

| Row | Col | Value | Purpose |
|-----|-----|-------|---------|
| 7 | 6 | 1.0 | X7 accumulates X6 |
| 7 | 7 | 1.0 | X7 self-recurrence |
| 10 | 10 | 1.0 | X10 position self-recurrence |
| 11 | 11 | 1.0 | X11 position self-recurrence |
| 13 | 13 | 1.0 | X13 correction latch |
| 16 | 15 | 1000.0 | X16 armed from X15 |
| 16 | 16 | 10.0 | X16 self-recurrence |
| 17 | 15 | -100000.0 | X17 subtract delayed energy |
| 17 | 16 | 1000.0 | X17 arm gate |
| 18 | 17 | 10.0 | X18 driven by X17 |
| 18 | 18 | 10.0 | X18 self-recurrence |
| 19 | 19 | 1.0 | * X19 counter self-recurrence |
| 20 | 20 | 1.0 | * X20 steering self-recurrence |
| 21 | 21 | 1.0 | * X21 corridor-active self-recurrence |
| 22 | 22 | 1.0 | * X22 countdown self-recurrence |

### W_out (1 x n) — non-zero entries

New entry for v2 marked with *.

| Col | Value | Neuron |
|-----|-------|--------|
| 0 | -0.229 | X0: hit turn |
| 1 | -0.698 | X1: heading (left) |
| 2 | +0.698 | X2: heading (right) |
| 3 | -0.349 | X3: safety left |
| 4 | +0.349 | X4: safety right |
| 5 | -0.349 | X5: front-block |
| 20 | 1.0 | * X20: shortcut steering override |


## Activation function per neuron

| Neurons | Activation | Notes |
|---------|------------|-------|
| X0-X5 | `relu_tanh` | Feature extraction |
| X6 | Custom | `clip(O_features + X20, -a, a)` |
| X7 | Identity | Pass-through of accumulated dtheta |
| X8-X9 | Custom | `cos/sin(theta_now)` |
| X10-X11 | Custom | Gated integrator (freeze on hit) |
| X12 | `relu_tanh` | Hit signal for gating |
| X13 | Custom | Sample-and-hold correction |
| X14 | Identity | Pass-through |
| X15-X18 | `relu_tanh` | Reward circuit |
| X19 | Custom | Step counter (cooldown) |
| X20 | Custom | Steering override (three-mode) |
| X21 | Custom | Corridor-active countdown |
| X22 | Custom | Phase countdown timer |
| X23-X999 | `relu_tanh` | Unused (all zero) |


## Constants

| Name | Symbol | Value | Description |
|------|--------|-------|-------------|
| n | -- | 1000 | Total neurons |
| p | -- | 64 | Camera resolution (rays) |
| speed | v | 0.01 | Bot speed (units/step) |
| leak | -- | 1.0 | Full state replacement |
| warmup | -- | 0 | No warmup steps |
| Shortcut turn | S | -2.0 | Turn magnitude (saturates clamp) |
| Horizontal threshold | tau_horiz | 0.35 | `|sin|` below this = horizontal |
| Vertical threshold | tau_vert | 0.70 | `|sin|` above this = vertical |
| Cooldown threshold | N_cooldown | 60 | Min steps before re-trigger |
| Center threshold | delta_center | 0.05 | Position tolerance for trigger |
| Drift offset | D | 0.115 | Arc-radius compensation |
| Corridor gain | G | 5.0 | Centering correction gain |
| Corridor sin threshold | tau_corr | 0.30 | Min vertical heading for centering |
| Corridor steps | N_corridor | 120 | X21 active duration |
| Turn steps | N_turn | 18 | Duration of 90-deg turn |
| Approach steps | N_approach | 45 | Duration of straight traverse |
| Total shortcut | N_total | 63 | N_turn + N_approach |
| Calibration gain | c | 5.78 | 1/0.173, heading correction |
| Energy scale | K | 0.005 | Reward circuit scaling |


## Circuit diagram (signal flow)

```
INPUTS                    FEATURE NEURONS                    OUTPUT
-----------               ---------------                    ------
ray[20] -----> X1 (L) ---+
ray[43] -----> X2 (R) ---+
ray[11] -----> X3 (sL) --+--> O_features --+--> clip --> O (steering)
ray[52] -----> X4 (sR) --+                 |
ray[31,32] --> X5 (fr) --+                 |
hit ---------> X0 (ht) --+                 |
                                            |
                SHORTCUT CIRCUIT             |
                ----------------             |
                                             |
  X18 (rewarded) ----+                       |
  sin(theta_lag) ----+                       |
  X5 (front) --------+--> TRIGGER -----> X22 (countdown)
  X19 (counter) -----+                    |
  X10 (x-pos) -------+                    +--> X20 (override) --+
                                           |       |             |
                                           |   turn_toward      |
                                           |   O_no_front       |
                                           |                    |
                                           +--> X21 (corridor)  |
                                                  |             |
                                               centering        |
                                                                |
                 HEAD-DIRECTION TRACKING                        |
                 ----------------------                         |
            X0..X5 --> O_features + X20 --> dtheta (X6) --------+
                              |
                X6 --> X7 (direction accumulator)
                              |
       ray[20,43] --> X13 (sample-and-hold correction)
       ray[20,43] --> X14 (instantaneous correction)
                              |
            X7 + correction + dtheta + pi/2 = theta_now
                              |
                    X8 = cos(theta), X9 = sin(theta)
                              |
           hit --> X12 ----> gate
                              |
                X10 += v * cos  (if no hit)
                X11 += v * sin  (if no hit)

                 REWARD-STATE DETECTION
                 ----------------------
       energy --> X15 (delayed scaled copy)
                    |
               X15 --> X16 (armed latch)
                    |
       energy --> X17 (pulse: energy rise, gated by X16)
                    |
               X17 --> X18 (is_rewarded latch) -----> feeds TRIGGER
```


## Behavioral summary

| Phase | Behavior | Duration |
|-------|----------|----------|
| First lap | Full clockwise outer ring (identical to v1) | ~400-500 steps |
| First reward | X18 latches to 1.0; shortcut circuit arms | instant |
| Trigger | All 5 conditions met at corridor entrance | instant |
| Turn | Hard 90-deg turn at max rate into corridor | 18 steps |
| Approach | Straight-line corridor traverse (front-block only) | 45 steps |
| Resume | Normal wall-following on opposite side | until next trigger |
| Subsequent | Half-ring laps, alternating side + corridor | ~200-250 steps/lap |

The half-ring pattern approximately doubles the reward collection
rate compared to v1, since the bot visits a reward source every
half-lap instead of every full lap.
