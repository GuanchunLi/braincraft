# Reflex Player v3 -- Environment 2

## Overview

Reflex Player v3 is a hand-crafted recurrent neural network (RNN) controller for a navigating bot in Environment 2 of the Braincraft challenge. It extends Reflex Player v2 with a single behavioral change: the **front-block steering gain** is no longer a fixed negative value but is **latched from early color evidence** gathered during the first few time steps. This allows the bot to adaptively choose its turning direction when a frontal obstacle is detected, based on which side the blue wall appears on.

The bot performs clockwise wall-following around a ring-shaped arena, with a corridor shortcut that it learns to exploit after encountering its first reward. An initial heading correction phase compensates for random starting jitter.

---

## 1. Network Architecture

The model is a discrete-time recurrent neural network with the following components:

| Component | Symbol | Dimensions |
|-----------|--------|------------|
| Input weight matrix | $W_{\text{in}}$ | $1000 \times 131$ |
| Recurrent weight matrix | $W$ | $1000 \times 1000$ |
| Output weight matrix | $W_{\text{out}}$ | $1 \times 1000$ |
| Hidden state | $X$ | $1000 \times 1$ |
| Input vector | $I$ | $131 \times 1$ |
| Output | $O$ | scalar |

### 1.1 Input Vector

The input $I$ has $2p + 3 = 131$ dimensions (where $p = 64$ is the camera resolution):

$$
I = \begin{pmatrix} 1 - d_0 \\ \vdots \\ 1 - d_{63} \\ c_0 \\ \vdots \\ c_{63} \\ \text{hit} \\ \text{energy} \\ 1.0 \end{pmatrix}
$$

- $d_i \in [0, 1]$: raw depth from camera ray $i$ (inverted so closer = higher)
- $c_i$: color/material ID from camera ray $i$
- hit: 1 if the bot collided on the previous step, 0 otherwise
- energy: current energy level
- 1.0: constant bias term

### 1.2 State Update Rule

At each time step $t$, the hidden state is updated as:

$$
X^{(t)} = (1 - \lambda)\, X^{(t-1)} + \lambda\, f\!\Big(W_{\text{in}}\, I^{(t)} + W\, X^{(t-1)}\Big)
$$

where $\lambda = 1.0$ (leak rate), so the update simplifies to:

$$
X^{(t)} = f\!\Big(W_{\text{in}}\, I^{(t)} + W\, X^{(t-1)}\Big)
$$

The output is:

$$
O^{(t)} = W_{\text{out}}\, g\!\big(X^{(t)}\big)
$$

where $g$ is the identity function. The scalar output $O$ is a steering angle increment ($\Delta\theta$).

---

## 2. Activation Function $f$

The activation function $f$ is a **custom, per-neuron** function that applies different logic to different hidden units. Let $z = W_{\text{in}} I + W X^{(t-1)}$ be the pre-activation vector.

### 2.1 Default Activation: `relu_tanh`

Most neurons use a rectified hyperbolic tangent:

$$
\text{relu\_tanh}(z) = \max\!\big(0,\; \tanh(z)\big)
$$

### 2.2 Neuron Roles

The 1000 hidden units are partitioned by function. Only a small fraction are actively wired; the rest remain zero throughout.

| Neuron(s) | Name | Activation | Role |
|-----------|------|------------|------|
| $X_0$ | Hit | relu_tanh | Collision signal |
| $X_1$ | Prox-Left | relu_tanh | Left-forward proximity |
| $X_2$ | Prox-Right | relu_tanh | Right-forward proximity |
| $X_3$ | Safety-Left | relu_tanh | Left-side wall distance |
| $X_4$ | Safety-Right | relu_tanh | Right-side wall distance |
| $X_5$ | Front-Block | relu_tanh | Frontal obstacle detector |
| $X_6$ | $\Delta\theta$ | custom (clamped) | Current steering output |
| $X_7$ | Direction accumulator | identity | Cumulative heading $\sum \Delta\theta$ |
| $X_8, X_9$ | cos/sin | custom | $\cos(\theta), \sin(\theta)$ of current heading |
| $X_{10}, X_{11}$ | Position | custom (gated) | Estimated $(x, y)$ displacement |
| $X_{12}$ | Hit relay | relu_tanh | Hit signal for position gating |
| $X_{13}$ | Correction hold | custom | Latched heading correction |
| $X_{14}$ | Correction instantaneous | identity | Current sensor asymmetry |
| $X_{15}$--$X_{18}$ | Reward circuit | relu_tanh | Energy-pulse reward detector/latch |
| $X_{19}$ | Step counter | custom | Steps since last corridor crossing |
| $X_{20}$ | Steering override | custom | Shortcut/correction override |
| $X_{22}$ | Shortcut countdown | custom | Countdown for shortcut maneuver |
| $X_{23}$ | Init correction remainder | custom | Remaining initial heading fix |
| $X_{24}$ | Seeded flag | custom | One-shot flag for init correction |
| $X_{25}$--$X_{25+p-1}$ | Color copies | identity | Raw color channel copies |
| $X_{\text{left\_blue}}$ | Left-blue indicator | custom | Blue detected on left side |
| $X_{\text{right\_blue}}$ | Right-blue indicator | custom | Blue detected on right side |
| $X_{\text{evidence}}$ | Evidence accumulator | custom | Accumulated left/right blue evidence |
| $X_{\text{front\_sign}}$ | Front-sign latch | custom | Latched sign for front gain |
| $X_{\text{signed\_front}}$ | Signed front value | custom | $\text{sign} \times X_5$ |

**Index computation for v3-specific neurons** (where $p = 64$):

- Color copies: $X_{25}$ to $X_{25 + p - 1} = X_{88}$
- Left-blue indicator: $X_{89}$
- Right-blue indicator: $X_{90}$
- Evidence accumulator: $X_{91}$
- Front-sign latch: $X_{92}$
- Signed-front value: $X_{93}$

---

## 3. Feature Neurons (X0--X5)

### 3.1 Input Wiring

Each feature neuron $X_i$ receives a weighted combination of specific input channels:

$$
z_0 = I_{\text{hit}}
$$

$$
z_1 = I_{20} \quad (\text{depth ray at index 20, left-forward})
$$

$$
z_2 = I_{43} \quad (\text{depth ray at index 43, right-forward})
$$

$$
z_3 = -I_{11} + 0.75 \quad (\text{left-side safety, offset by target distance})
$$

$$
z_4 = -I_{52} + 0.75 \quad (\text{right-side safety, offset by target distance})
$$

$$
z_5 = I_{31} + I_{32} - 1.4 \quad (\text{front-block: sum of two central rays minus threshold})
$$

Each is passed through $\text{relu\_tanh}$:

$$
X_i^{(t)} = \text{relu\_tanh}(z_i), \quad i = 0, \ldots, 5
$$

### 3.2 Output Contribution

The wall-following steering signal from the six feature neurons is:

$$
O_{\text{features}} = \sum_{i=0}^{5} w_i \, X_i
$$

with weights:

| $i$ | Gain Name | Value |
|-----|-----------|-------|
| 0 | hit turn | $-10° / \tanh(1)$ |
| 1 | heading (left) | $-40°$ |
| 2 | heading (right) | $+40°$ |
| 3 | safety left | $-20°$ |
| 4 | safety right | $+20°$ |
| 5 | front block | $0°$ (zeroed in v3; replaced by signed front) |

All angles are in radians. In v3, the $w_5 = 0$ entry is replaced by the signed front-block mechanism described in Section 4.

---

## 4. Front-Block Sign Latch (v3 Extension)

This is the key difference from Reflex Player v2. Instead of a fixed negative front-block gain, v3 determines the sign from early observations of the blue wall.

### 4.1 Color Evidence Gathering

At each activation, the raw color values $c_0, \ldots, c_{p-1}$ are copied into $X_{25}, \ldots, X_{88}$ via $W_{\text{in}}$. The activation then checks for the presence of the blue wall (color value $= 4.0$) on each side:

$$
\text{left\_blue} = \exists\, j \in \{0, \ldots, 31\} : c_j \approx 4.0
$$

$$
\text{right\_blue} = \exists\, j \in \{32, \ldots, 63\} : c_j \approx 4.0
$$

From these, binary evidence signals are produced:

$$
L_{\text{ev}} = \begin{cases} 1 & \text{if left\_blue} \wedge \neg\text{right\_blue} \\ 0 & \text{otherwise} \end{cases}
$$

$$
R_{\text{ev}} = \begin{cases} 1 & \text{if right\_blue} \wedge \neg\text{left\_blue} \\ 0 & \text{otherwise} \end{cases}
$$

### 4.2 Evidence Accumulation and Latching

The evidence accumulator $X_{\text{evidence}}$ integrates over time. Let $s_{\text{prev}} = X_{\text{front\_sign}}^{(t-1)}$ be the previous sign latch value.

**If the sign has not yet been latched** ($|s_{\text{prev}}| < 0.5$):

$$
X_{\text{evidence}}^{(t)} = X_{\text{evidence}}^{(t-1)} + R_{\text{ev}} - L_{\text{ev}}
$$

$$
X_{\text{front\_sign}}^{(t)} = \begin{cases}
+1 & \text{if } X_{\text{evidence}}^{(t)} \geq +2 \\
-1 & \text{if } X_{\text{evidence}}^{(t)} \leq -2 \\
0 & \text{otherwise (not yet latched)}
\end{cases}
$$

The threshold of $\pm 2$ ensures the sign is only committed after consistent evidence across multiple frames.

**If the sign has been latched** ($|s_{\text{prev}}| \geq 0.5$):

$$
X_{\text{evidence}}^{(t)} = X_{\text{evidence}}^{(t-1)} \quad (\text{frozen})
$$

$$
X_{\text{front\_sign}}^{(t)} = s_{\text{prev}} \quad (\text{held})
$$

### 4.3 Signed Front-Block Contribution

The signed front value is:

$$
X_{\text{signed\_front}}^{(t)} = X_{\text{front\_sign}}^{(t)} \times X_5^{(t)}
$$

The total feature-based steering becomes:

$$
O_{\text{features}} = \underbrace{\sum_{i=0}^{4} w_i \, X_i}_{\text{heading + safety}} + \underbrace{\alpha \cdot X_{\text{signed\_front}}}_{\text{adaptive front-block}}
$$

where $\alpha = 20°$ (in radians, $\approx 0.349$) is the front gain magnitude (`FRONT_GAIN_MAG`).

**Interpretation**: If the blue wall is on the right, the evidence accumulator goes positive, latching $+1$. The front-block then steers the bot to the right when facing a wall. If blue is on the left, the latch is $-1$ and the bot steers left. Before the latch fires, the front-block term is zero (the bot relies purely on heading and safety gains).

---

## 5. Heading Estimation

### 5.1 Correction Term

The heading correction compensates for sensor-to-heading misalignment. On the very first step ($X_7 \approx 0$):

$$
\text{correction} = X_{13}^{(t-1)}
$$

On subsequent steps:

$$
\text{correction} = X_{13}^{(t-1)} - X_{14}^{(t-1)}
$$

where $X_{13}$ is seeded from the initial sensor asymmetry via:

$$
X_{13} \leftarrow X_{13}^{\text{prev}} + \frac{I_{43} - I_{20}}{0.173}
$$

and $X_{14}$ tracks the instantaneous (non-recurrent) part of the same signal.

### 5.2 Lagged and Current Heading

The **lagged heading** (before the current steering update):

$$
\theta_{\text{lag}} = X_7^{(t-1)} + \frac{\pi}{2} + \text{correction}
$$

The **current heading** (after applying the steering output):

$$
\theta_{\text{now}} = X_7^{(t-1)} + \frac{\pi}{2} + \text{correction} + \Delta\theta^{(t)}
$$

### 5.3 Direction and Position Tracking

Direction components:

$$
X_8^{(t)} = \cos(\theta_{\text{now}}), \quad X_9^{(t)} = \sin(\theta_{\text{now}})
$$

Position integration (gated by collision):

$$
X_{10}^{(t)} = \begin{cases}
X_{10}^{(t-1)} + v \cos(\theta_{\text{now}}) & \text{if no hit} \\
X_{10}^{(t-1)} & \text{if hit}
\end{cases}
$$

$$
X_{11}^{(t)} = \begin{cases}
X_{11}^{(t-1)} + v \sin(\theta_{\text{now}}) & \text{if no hit} \\
X_{11}^{(t-1)} & \text{if hit}
\end{cases}
$$

where $v = 0.01$ is the bot speed.

---

## 6. Initial Heading Correction

The bot starts with direction $\frac{\pi}{2} + \epsilon$, where $\epsilon \sim U(-5°, +5°)$. The initial correction circuit physically turns the bot back to exactly upward.

### 6.1 Seeding (First Activation, $X_{24} < 0.5$)

$$
X_{23}^{(t)} = \text{clip}\!\big(-\text{correction},\; -0.15,\; 0.15\big)
$$

$$
X_{24}^{(t)} = 1.0
$$

No steering override is applied (output from this step is discarded by warmup).

### 6.2 Draining (Subsequent Steps, $X_{24} \geq 0.5$)

While $|X_{23}| > 10^{-3}$:

$$
\Delta\theta_{\text{init}} = \text{clip}\!\big(X_{23},\; -a,\; a\big), \quad a = 5°
$$

$$
X_{23}^{(t)} = X_{23}^{(t-1)} - \Delta\theta_{\text{init}}
$$

The steering override is set to cancel wall-following and apply only the correction:

$$
X_{20} = \Delta\theta_{\text{init}} - O_{\text{features}}
$$

---

## 7. Reward Detection Circuit (X15--X18)

This circuit detects when the bot has collected a reward (energy increase).

$$
X_{15}^{(t)} = \text{relu\_tanh}\!\big(0.005 \cdot I_{\text{energy}}\big)
$$

$$
X_{16}^{(t)} = \text{relu\_tanh}\!\big(1000 \cdot X_{15}^{(t-1)} + 10 \cdot X_{16}^{(t-1)}\big)
$$

$$
X_{17}^{(t)} = \text{relu\_tanh}\!\big(500 \cdot I_{\text{energy}} - 100000 \cdot X_{15}^{(t-1)} + 1000 \cdot X_{16}^{(t-1)} - 1000.2\big)
$$

$$
X_{18}^{(t)} = \text{relu\_tanh}\!\big(10 \cdot X_{17}^{(t-1)} + 10 \cdot X_{18}^{(t-1)}\big)
$$

$X_{18}$ acts as a permanent latch: once a reward pulse is detected ($X_{17}$ fires briefly when energy increases), $X_{18}$ saturates to 1 and stays there due to its self-recurrence. The condition $X_{18} > 0.5$ is used as a "has been rewarded" flag.

---

## 8. Shortcut Circuit

After the first reward is detected, the bot attempts to take a corridor shortcut on subsequent laps instead of going around the full ring.

### 8.1 Step Counter ($X_{19}$)

$$
X_{19}^{(t)} = \begin{cases}
0 & \text{if } |X_{10}^{(t-1)}| < 0.05 \text{ and } |\sin(\theta_{\text{lag}})| > 0.70 \\
X_{19}^{(t-1)} + 1 & \text{otherwise}
\end{cases}
$$

The counter resets when the bot is near the center ($x \approx 0$) and heading vertically (i.e., traversing the corridor).

### 8.2 Shortcut Trigger

The shortcut fires when **all** of these conditions hold simultaneously:

1. $X_{18} > 0.5$ (reward has been seen)
2. $|\sin(\theta_{\text{lag}})| < 0.35$ (heading is approximately horizontal)
3. $X_5 < 0.1$ (no frontal obstacle)
4. $X_{19} > 60$ (enough steps since last corridor)
5. $|X_{10} - \delta| < 0.05$ where $\delta = -0.115 \cdot \text{sign}(\cos\theta_{\text{lag}})$ (near corridor entrance, arc-corrected)
6. $X_{22} < 0.5$ (not already in a shortcut)

### 8.3 Shortcut Execution ($X_{22}$ Countdown)

When triggered, $X_{22}$ is set to $63$ (= 18 turn steps + 45 approach steps) and decrements by 1 each step.

**Turning phase** ($X_{22} > 45.5$): The bot executes a sharp turn into the corridor:

$$
X_{20} = 2.0 \times d_{\text{turn}}
$$

where $d_{\text{turn}} = -\text{sign}(\cos\theta_{\text{lag}}) \cdot \text{sign}(X_{11})$ determines the turn direction (the value saturates at the $\pm 5°$ clamp).

**Approach phase** ($0.5 < X_{22} \leq 45.5$): The bot drives straight, canceling all feature contributions except the front-block:

$$
X_{20} = -O_{\text{no\_front}}, \quad O_{\text{no\_front}} = \sum_{i=0}^{4} w_i X_i
$$

---

## 9. Final Output Computation

The final steering angle per time step is computed as:

$$
O_{\text{now}} = O_{\text{features}} + X_{20}
$$

$$
\Delta\theta^{(t)} = \text{clip}\!\big(O_{\text{now}},\; -5°,\; +5°\big)
$$

This clamped value is stored in $X_6$ and fed back through the recurrence into $X_7$ (direction accumulator) at the next step:

$$
X_7^{(t)} = X_7^{(t-1)} + X_6^{(t)}
$$

The output scalar $O = W_{\text{out}} X$ drives the bot's `forward()` method, which applies the steering angle and advances the bot by one step.

---

## 10. Summary of v3 vs v2

| Aspect | Reflex v2 | Reflex v3 |
|--------|-----------|-----------|
| Front-block sign | Fixed $-20°$ | Latched from blue-wall evidence |
| Color channels | Unused | Read during early steps |
| Additional neurons | -- | 5 extra (color copies, evidence, sign latch, signed output) |
| Behavior | Always turns same direction at walls | Adapts turn direction to arena layout |

The front-sign latch makes v3 robust to arena configurations where the optimal turn direction depends on which side the blue wall is on -- information that v2 ignores entirely.
