# `env1_player_dummy3.py` - Model Structure

A deterministic, hand-wired Task-1 controller built inside the standard
1000-neuron Braincraft RNN interface. Like the other dummy players, it uses a
very sparse reservoir: only a small set of neurons carry any signal, and the
rest exist only to satisfy the challenge contract.

Compared with the simpler dummy players, `dummy3` adds an explicit internal
reward-state subcircuit:

- `X9` is the persistent `is_rewarded` flag.
- `X8` is a transient pulse that marks the first observable positive energy
  jump.
- `X6` and `X7` are helper neurons used to build that pulse robustly.
- `X10` is a post-reward detector gated by `X9`.

Important implementation detail: in the current checked-in file,
`Wout[0,10] = 0`, so the reward-state circuit is computed and can be plotted,
but it does **not yet affect steering**. The actual motion law currently comes
from the hit, safety, and front-block neurons only.

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

For `dummy3`:

- $n = 1000$
- $p = 64$
- $\lambda = 1.0$
- `warmup = 0`
- $f = \texttt{relu\_tanh}$
- $g = \texttt{identity}$

Since `leak = 1`, the update simplifies to:

$$
\mathbf{x}_{t+1}
=
f\!\left(W_{\text{in}}\mathbf{i}_t + W\mathbf{x}_t\right)
$$

The output is passed to `bot.forward(...)`, which clips the actual turn to
`[-5 deg, +5 deg]` per step.

---

## 2. Inputs and nonlinearity

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
$$

where:

- `1 - d_{t,k}` is ray proximity: larger means closer.
- input index `64` is `hit`
- input index `65` is `energy`
- input index `66` is a constant bias `1`

Define:

$$
\text{prox}_{t,k} \equiv 1 - d_{t,k}.
$$

The hidden nonlinearity is

$$
\rho(z) \equiv \max(\tanh(z), 0),
$$

which is exactly the code's `relu_tanh`.

This matters because:

- positive inputs pass through approximately linearly when they are small,
- negative inputs are clamped to zero,
- large positive inputs saturate near `1`.

---

## 3. Sparse state layout

Only the first 11 neurons are used:

| neuron | role | contributes to output now? |
|---|---|---|
| `X0` | hit detector | yes |
| `X1` | left symmetric-front ray | no, because `heading_gain = 0` |
| `X2` | right symmetric-front ray | no, because `heading_gain = 0` |
| `X3` | left safety feature | yes |
| `X4` | right safety feature | yes |
| `X5` | front-block detector | yes |
| `X6` | delayed energy copy | no |
| `X7` | arm latch for reward detection | no |
| `X8` | transient reward pulse | no |
| `X9` | persistent `is_rewarded` latch | no |
| `X10` | post-reward outermost-ray asymmetry detector | no, because `Wout[0,10] = 0` |

All neurons `X11` through `X999` remain identically zero.

---

## 4. Closed-form state update

Let:

- `L_idx = 20`
- `R_idx = 43`
- `left_side_idx = 11`
- `right_side_idx = 52`
- center rays `C1 = 31`, `C2 = 32`

Then the non-zero feature updates are:

$$
x_{t+1,0} = \rho(\text{hit}_t)
$$

$$
x_{t+1,1} = \rho(\text{prox}_{t,20})
$$

$$
x_{t+1,2} = \rho(\text{prox}_{t,43})
$$

$$
x_{t+1,3} = \rho(\text{prox}_{t,11} - 0.7)
$$

$$
x_{t+1,4} = \rho(\text{prox}_{t,52} - 0.7)
$$

$$
x_{t+1,5} = \rho(\text{prox}_{t,31} + \text{prox}_{t,32} - 1.4)
$$

The reward-state block is:

$$
x_{t+1,6} = \rho(0.005\,E_t)
$$

$$
x_{t+1,7} = \rho(1000\,x_{t,6} + 10\,x_{t,7})
$$

$$
x_{t+1,8}
=
\rho(500\,E_t - 100000\,x_{t,6} + 1000\,x_{t,7} - 1000.2)
$$

$$
x_{t+1,9} = \rho(10\,x_{t,8} + 10\,x_{t,9})
$$

$$
x_{t+1,10}
=
\rho(-3\,\text{prox}_{t,0} + 3\,\text{prox}_{t,63} + 10\,x_{t,9} - 10)
$$

where $E_t = \text{energy}_t$ is the energy loaded into the input vector on
loop iteration $t$.

---

## 5. The current steering law

The readout gains in the current file are:

| term | code name | value |
|---|---|---|
| hit term | `hit_turn` | `np.radians(-10.0) / tanh(1)` |
| heading term | `heading_gain` | `0` |
| left safety | `safety_gain_left` | `np.radians(-20.0)` |
| right safety | `safety_gain_right` | `np.radians(+20.0)` |
| front block | `front_gain` | `np.radians(-20.0)` |
| post-reward term | `post_reward_gain` | `0.0 * front_gain = 0` |

So the actual output is

$$
o_t
=
\text{hit\_turn}\,x_{t+1,0}
+
\text{safety\_gain\_left}\,x_{t+1,3}
+
\text{safety\_gain\_right}\,x_{t+1,4}
+
\text{front\_gain}\,x_{t+1,5}.
$$

Equivalently:

$$
o_t
=
\frac{-10^\circ}{\tanh(1)}\,x_{t+1,0}
-20^\circ\,x_{t+1,3}
+20^\circ\,x_{t+1,4}
-20^\circ\,x_{t+1,5}
$$

interpreting the degree constants above as radians via `np.radians(...)`.

Two consequences are worth stating explicitly:

- `X1` and `X2` are computed but unused because `heading_gain = 0`.
- `X10` is computed but unused because `post_reward_gain = 0`.

So the reward-state machinery is presently an internal latent variable /
diagnostic channel, not a behavioral branch.

---

## 6. Why reward detection needs timing care

The subtlety comes from when energy is observed versus when it is updated.

In [challenge_1.py](challenge_1.py), each loop iteration does:

1. read the current bot state into `I`
2. update `X`
3. compute output `O`
4. call `bot.forward(O, environment, ...)`

Inside [environment_1.py](environment_1.py), the energy source refill and the
movement / hit penalties are applied during `environment.update(bot)`:

$$
\text{bot.energy} \leftarrow \text{bot.energy} + \text{refill}
$$

then

$$
\text{bot.energy} \leftarrow \text{bot.energy} - \text{energy\_move}
$$

and possibly

$$
\text{bot.energy} \leftarrow \text{bot.energy} - \text{energy\_hit}.
$$

Therefore:

- the controller sees `energy_t` **before** the move,
- any refill that happens during that move is only visible as `energy_{t+1}`
  on the next loop iteration.

This is why the circuit is detecting the first **observable positive jump in
the input energy trace**, not source occupancy by itself.

---

## 7. Reward-state circuit in detail

### 7.1 `X6`: delayed energy copy

`X6` stores a compressed copy of the current energy:

$$
x_{t+1,6} = \rho(0.005\,E_t).
$$

Since `energy` is typically in `[0,1]`, the argument is in `[0,0.005]`, so
`tanh(z) ~ z` and thus

$$
x_{t+1,6} \approx 0.005\,E_t.
$$

So on the next step, `X6` behaves like a one-step delayed energy sample.

### 7.2 `X7`: arm latch

`X7` is not the reward flag. It is a startup guard:

$$
x_{t+1,7} = \rho(1000\,x_{t,6} + 10\,x_{t,7}).
$$

At `t=0`, both `x_{0,6}` and `x_{0,7}` are zero, so `X7` is zero.
After one valid energy sample has been written into `X6`, the term
`1000 * x_{t,6}` is already large and positive, so `X7` rises near `1` and
then self-sustains.

Interpretation:

- before `X7` turns on, the pulse detector is disabled,
- after `X7` turns on, the pulse detector is allowed to compare consecutive
  energy samples.

This is what prevents the old startup bug where the very first energy sample
looked like a reward simply because the delayed copy was still zero.

### 7.3 `X8`: transient reward pulse

`X8` computes

$$
x_{t+1,8}
=
\rho(500\,E_t - 100000\,x_{t,6} + 1000\,x_{t,7} - 1000.2).
$$

Once the arm is on (`x_{t,7} ~ 1`) and the delayed copy is near-linear
(`x_{t,6} ~ 0.005\,E_{t-1}`), this becomes approximately

$$
x_{t+1,8}
\approx
\rho(500\,(E_t - E_{t-1}) - 0.2).
$$

So `X8` is a thresholded detector of positive net energy jumps:

- normal drain: `E_t - E_{t-1} ~ -0.001`

  $$
  500(-0.001) - 0.2 = -0.7
  $$

  so `X8 = 0`.

- typical refill without a hit: net rise `~ +0.004`

  $$
  500(+0.004) - 0.2 = 1.8
  $$

  so `X8` becomes strongly positive and saturates near `1`.

- startup: `x_{t,7} = 0`, `x_{t,6} = 0`, and `E_t <= 1`

  $$
  500\,E_t - 1000.2 \le -500.2
  $$

  so the startup sample cannot trigger a false pulse.

This is the key robustness improvement in the revised implementation.

### 7.4 `X9`: persistent `is_rewarded` latch

The actual reward-state variable is:

$$
x_{t+1,9} = \rho(10\,x_{t,8} + 10\,x_{t,9}).
$$

This is a standard self-exciting latch:

- before the first reward pulse, both inputs are zero, so `X9 = 0`,
- when `X8` pulses, the next `X9` update receives a large positive drive,
- after `X9` has risen, the self-connection keeps it pinned near `1`.

So `X9` means:

> "Has the controller ever observed a positive net energy increase so far?"

That is the correct interpretation of the current `is_rewarded` state.

### 7.5 `X10`: gated post-reward detector

The last neuron in the reward block is

$$
x_{t+1,10}
=
\rho(-3\,\text{prox}_{t,0} + 3\,\text{prox}_{t,63} + 10\,x_{t,9} - 10).
$$

Before reward, `x_{t,9} = 0`, so the argument is at most

$$
-3 \cdot 0 + 3 \cdot 1 - 10 = -7,
$$

which means `X10 = 0` always.

After reward, `x_{t,9} ~ 1`, so the gating terms approximately cancel:

$$
x_{t+1,10}
\approx
\rho(3(\text{prox}_{t,63} - \text{prox}_{t,0})).
$$

So after the latch is set, `X10` becomes a thresholded detector of whether
the rightmost ray is closer than the leftmost ray. In the current file that
signal is computed but does not steer the bot because:

$$
W_{\text{out}}[0,10] = 0.
$$

---

## 8. Timeline of the first reward event

Suppose the bot enters the source and, after refill minus costs, its energy
increases during the move executed on loop iteration `t`.

Then:

1. loop `t` still used the old energy `E_t` when computing `X`
2. after `bot.forward(...)`, the bot now has higher energy
3. loop `t+1` sees the higher input energy `E_{t+1} > E_t`
4. on loop `t+1`, `X8` pulses
5. on loop `t+2`, `X9` latches high

So:

- `X8` is the "reward just happened" pulse,
- `X9` is the persistent "reward has happened sometime in the past" state.

If no positive net energy rise ever occurs, then `X8`, `X9`, and `X10` all
stay at zero for the entire rollout.

---

## 9. Practical debugging notes

For plotting / debugging:

- `X[8,0]` is the transient reward pulse
- `X[9,0]` is the persistent `is_rewarded` state
- `X[10,0]` is the post-reward gated detector

The plotting helper [env1_trajectory_variable_plot.py](env1_trajectory_variable_plot.py)
now defaults to:

```python
DEFAULT_VAR = "X[9,0]"
```

which is the right variable to inspect when you want to see the internal
reward-state latch.

---

## 10. File map

| file | role |
|---|---|
| [env1_player_dummy3.py](env1_player_dummy3.py) | current sparse controller and reward-state circuit |
| [env1_trajectory_variable_plot.py](env1_trajectory_variable_plot.py) | trajectory + state plotting, defaulting to `X[9,0]` |
| [challenge_1.py](challenge_1.py) | rollout loop and RNN update order |
| [environment_1.py](environment_1.py) | refill timing and movement / hit energy costs |
| [bot.py](bot.py) | turn clipping and movement |

