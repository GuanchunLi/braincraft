# `env1_player_dummy.py` — Model Structure

A deterministic, hand-wired controller that drives the bot clockwise around
the outer ring of [environment_1.py](environment_1.py) by following the wall
on its **left** side. It is built as a 1000-neuron feed-forward instance of
the standard Braincraft RNN, but only **3 neurons carry signal** — the rest
exist solely to satisfy the 1000-neuron interface.

---

## 1. The generic Braincraft RNN

Every Task-1 player is evaluated by [challenge_1.py](challenge_1.py) using
the same update rule (see `evaluate()` lines 145–159):

$$
\mathbf{x}_{t+1} \;=\; (1 - \lambda)\,\mathbf{x}_t \;+\; \lambda\,
f\!\left(W_{\text{in}}\,\mathbf{i}_t \;+\; W\,\mathbf{x}_t\right)
$$

$$
o_t \;=\; W_{\text{out}}\,g(\mathbf{x}_{t+1})
$$

with

| symbol            | shape         | meaning                                          |
|-------------------|---------------|--------------------------------------------------|
| $\mathbf{i}_t$    | $(p+3,1)$     | sensor input at step $t$                         |
| $\mathbf{x}_t$    | $(n,1)$       | hidden state ($n=1000$)                          |
| $W_{\text{in}}$   | $(n,p+3)$     | input weights                                    |
| $W$               | $(n,n)$       | recurrent weights                                |
| $W_{\text{out}}$  | $(1,n)$       | readout                                          |
| $\lambda$         | scalar        | leak (1.0 here → no temporal integration)        |
| $f,\,g$           | element-wise  | hidden / readout nonlinearities                  |
| $o_t$             | scalar        | desired heading change in **radians**            |

The output $o_t$ is then passed to `bot.forward(o_t, …)` which clamps it to
the per-step turn limit:

$$
\Delta\theta_t \;=\; \mathrm{clip}\!\left(o_t,\;-\tfrac{5\pi}{180},\;
+\tfrac{5\pi}{180}\right)
\qquad\Longleftrightarrow\qquad |\Delta\theta_t|\le 5^\circ
$$

After the heading update $\theta_{t+1} = \theta_t + \Delta\theta_t$, the bot
takes one fixed-length step:

$$
\mathbf{p}_{t+1} \;=\; \mathbf{p}_t \;+\; v\,\bigl(\cos\theta_{t+1},\,
\sin\theta_{t+1}\bigr),\qquad v=0.01.
$$

**Sign convention** (math angles, $y$ axis up):

- $o_t > 0\;\Longrightarrow\;\Delta\theta>0$: counter-clockwise turn (**left**)
- $o_t < 0\;\Longrightarrow\;\Delta\theta<0$: clockwise turn (**right**)

---

## 2. The input vector $\mathbf{i}_t$

From [challenge_1.py:150-152](challenge_1.py#L150-L152):

$$
\mathbf{i}_t \;=\;
\begin{bmatrix}
1 - d_{t,0} \\[2pt]
1 - d_{t,1} \\[2pt]
\vdots \\[2pt]
1 - d_{t,p-1} \\[4pt]
\text{hit}_t \\[2pt]
\text{energy}_t \\[2pt]
1
\end{bmatrix}
\quad\in\;\mathbb{R}^{p+3},\qquad p=64.
$$

- $d_{t,k}\in[0,1]$ is the depth of camera ray $k$ (0 = touching, 1 = far),
  so $1 - d_{t,k}$ is **proximity** — higher means closer.
- Camera ray $k=0$ is the **leftmost** ray, $k=63$ the **rightmost** (see
  [camera.py:137](camera.py#L137); the linspace runs $+1\to-1$ and
  $\theta_{\text{ray}}=\theta+\arctan(X/D)$, so positive $X$ adds CCW
  rotation).
- Index $p=64$ is the binary collision flag, $p+1=65$ is the energy gauge,
  $p+2=66$ is a constant bias of $1$.

For convenience, define for the ray we actually use:

$$
s_t \;\equiv\; 1 - d_{t,11}
\qquad\text{(``side proximity'' on the bot's left)}.
$$

Index 11 is well inside the left half of the FOV but not at the extreme
edge, which keeps it from clipping during sharp corner turns.

---

## 3. The model

### 3.1 Hyperparameters

| symbol  | value      | source                                       |
|---------|------------|----------------------------------------------|
| $n$     | $1000$     | required by challenge                        |
| $p$     | $64$       | `bot.camera.resolution`                      |
| $\lambda$ | $1$      | full overwrite (no recurrent integration)    |
| $f$     | identity   | linear hidden activation                     |
| $g$     | identity   | linear readout                               |
| warmup  | $0$        | controller is active from $t=0$              |

With $\lambda=1$ and $W=0$ (see below), the recurrence collapses to a
**stateless feed-forward map**:

$$
\mathbf{x}_{t+1} \;=\; W_{\text{in}}\,\mathbf{i}_t,
\qquad
o_t \;=\; W_{\text{out}}\,\mathbf{x}_{t+1}
\;=\; (W_{\text{out}} W_{\text{in}})\,\mathbf{i}_t .
$$

### 3.2 Weight matrices

The recurrent matrix is identically zero:

$$
W \;=\; \mathbf{0}_{n\times n}.
$$

The input matrix has only three non-zero rows:

$$
W_{\text{in}}[0,\;p\,] \;=\; 1
\qquad\text{(row 0 reads }\text{hit}_t\text{)}
$$

$$
W_{\text{in}}[1,\;p+2\,] \;=\; 1
\qquad\text{(row 1 reads the bias }1\text{)}
$$

$$
W_{\text{in}}[2,\;11\,] \;=\; 1
\qquad\text{(row 2 reads }s_t = 1-d_{t,11}\text{)}
$$

so the only non-zero entries of the hidden state are

$$
x_{t+1,0} = \text{hit}_t,\qquad
x_{t+1,1} = 1,\qquad
x_{t+1,2} = s_t,
$$

and $x_{t+1,k}=0$ for all $k\ge 3$.

The readout has three non-zero entries:

$$
W_{\text{out}}[0,0] \;=\; H,\qquad
W_{\text{out}}[0,1] \;=\; -G\,s^{\star},\qquad
W_{\text{out}}[0,2] \;=\; G,
$$

with the three hand-tuned constants

| constant       | code name      | value   | role                              |
|----------------|----------------|---------|-----------------------------------|
| $H$            | `hit_turn`     | $-5.0$  | turn rate kick on collision       |
| $G$            | `wall_gain`    | $-0.40$ | proportional gain on the wall error |
| $s^{\star}$    | `wall_target`  | $+0.65$ | target left-side proximity        |

### 3.3 The closed-form control law

Combining the readout entries above:

$$
\boxed{\;
o_t \;=\; H\cdot\text{hit}_t \;+\; G\cdot(s_t - s^{\star})
\;}
$$

That is the *entire* controller. Plugging in the numbers:

$$
o_t \;=\; -5.0\;\text{hit}_t \;-\; 0.40\,(s_t - 0.65).
$$

Equivalently, expanding and folding the constant into a bias term:

$$
o_t \;=\; -5.0\;\text{hit}_t \;-\; 0.40\,s_t \;+\; 0.26 .
$$

After the bot's $\pm 5^\circ$ clamp:

$$
\Delta\theta_t \;=\; \mathrm{clip}\!\bigl(o_t,\;-0.0873,\;+0.0873\bigr)
\quad\text{rad}.
$$

---

## 4. What the law actually does

### 4.1 Free flight, no contact

When $\text{hit}_t=0$ the law reduces to a P-controller on the left-side
proximity error $e_t \equiv s_t - s^{\star}$:

$$
o_t \;=\; G\cdot e_t \;=\; -0.40\,(s_t - 0.65).
$$

| situation                  | $s_t$        | $e_t$    | $o_t$    | $\Delta\theta_t$       | meaning                  |
|----------------------------|--------------|----------|----------|------------------------|--------------------------|
| left wall too close        | $s_t > 0.65$ | $> 0$    | $< 0$    | clockwise (right)      | steer **away** from wall |
| left wall at target offset | $s_t = 0.65$ | $= 0$    | $= 0$    | go straight            | cruise                   |
| left wall too far          | $s_t < 0.65$ | $< 0$    | $> 0$    | counter-clockwise (left) | steer **toward** wall    |

Because $G$ is signed (not split into two ReLU half-neurons), **both**
directions of the error provide corrective torque. With $|e_t|\le 0.65$ the
unclamped output stays in $[-0.26,+0.26]$ rad, which exceeds the $0.087$ rad
clamp — so a saturated step happens whenever $|e_t|>0.218$, i.e. whenever
$s_t<0.43$ or $s_t>0.87$. In practice the bot rapidly snaps onto a thin band
around $s_t=s^{\star}=0.65$ and then makes only small per-step corrections.

The fixed point of the position dynamics is the orbit on which (i) the
left-ray sees a wall at depth $1-s^{\star}=0.35$ and (ii) the bot is moving
parallel to that wall. Since ray 11 is offset from the heading by

$$
\alpha_{11} \;=\; \arctan\!\frac{X_{11}}{D},\qquad
X_{11} = \frac{W}{2}\,\frac{63 - 2\cdot 11}{63},\quad
W = 2D\tan(30^\circ),\quad D=0.25,
$$

the bot is held at perpendicular distance $\approx (1-s^{\star})\sin\alpha_{11}$
from the wall. With $D=0.25$ and the camera FOV $=60^\circ$ this is a small
constant offset comfortably larger than the bot radius $0.05$.

### 4.2 Corner handling without an explicit "front" detector

Approaching a 90° corner (wall on the left, new wall straight ahead),
the left ray $k=11$ sweeps from the side wall onto the **front** wall as the
bot advances. The front wall is closer than the side wall was, so $s_t$
sharply increases past $s^{\star}$, $e_t$ becomes large and positive, and
$o_t$ saturates to $-5^\circ$/step CW. The bot pivots right until the new
wall on its left registers at the target proximity, then cruise resumes.
No separate "front_turn" neuron is needed.

### 4.3 Collision reflex

If the bot does touch a wall, $\text{hit}_t = 1$ and

$$
o_t \;=\; -5.0 \;+\; G\cdot e_t \;\le\; -5.0 + 0.26 \;<\; -0.087,
$$

so $\Delta\theta_t = -5^\circ$ regardless of the wall-error term. The bot
spins clockwise at the maximum rate until it is no longer in contact, then
the proportional law takes over.

### 4.4 Acquiring the ring from the start position

The bot starts at $(0.5,0.5)$ facing $\theta_0\approx 90^\circ$ (north),
i.e. **inside** the central column between the four pillars of
[environment_1.py](environment_1.py). Initially the left ray sees the
nearest pillar (col 3) at moderate distance $\Rightarrow s_0$ is below
$s^{\star}\Rightarrow o_0>0\Rightarrow$ slow CCW turn. The bot drifts left
until it acquires a wall at the target offset, after which the same
proportional law that handles the outer ring also keeps it on a stable
orbit. By the empirical trajectory check this orbit *is* the outer ring:
seeds 12345/1/2/7 all complete the full 1402-step energy budget with
**zero** wall hits and a total path length of $\approx 14.0$.

---

## 5. Why the previous controller failed (for contrast)

The earlier version used $f=$ `relu_tanh` and four feature neurons
implementing a **symmetric** "corridor centerer":

$$
o_{\text{old}} \;=\;
-0.28\,\mathrm{ReLU}\!\bigl(\tanh(F-0.58)\bigr)
\;+\;0.06\,\mathrm{ReLU}\!\bigl(\tanh(R-L)\bigr)
\;-\;0.06\,\mathrm{ReLU}\!\bigl(\tanh(L-R)\bigr)
\;-\;0.24\,\text{hit}_t
$$

where $L=\overline{1-d_{[0:16]}}$, $R=\overline{1-d_{[48:64]}}$,
$F=\overline{1-d_{[24:40]}}$. The $L,R$ terms try to enforce $L=R$ — i.e.
to *center* the bot between symmetric walls. On the outer ring there is
only one wall, so this term **pulls the bot off the ring** into the
interior. The new law uses a single signed P-controller on one side and
trivially does the right thing.

---

## 6. File map

| file                                              | what it provides                                          |
|---------------------------------------------------|-----------------------------------------------------------|
| [env1_player_dummy.py](env1_player_dummy.py)      | `dummy_player()` builds the model described here          |
| [challenge_1.py](challenge_1.py)                  | RNN update rule (`evaluate`, lines 145–159)               |
| [bot.py](bot.py)                                  | $\pm 5^\circ$ clamp (`forward`, line 138); kinematics     |
| [camera.py](camera.py)                            | depth ray ordering (`update`, line 137)                   |
| [environment_1.py](environment_1.py)              | 10×10 world with outer wall and four interior pillars     |
| [trajectory_dummy.py](trajectory_dummy.py)        | replays the controller and plots the trajectory           |
| [env3_player_wallfollow.py](env3_player_wallfollow.py) | the original right-wall version this player mirrors  |
