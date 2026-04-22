# env2_player_bio.py — numpy 2.x NEAR_CR gate hardening

**Date:** 2026-04-21
**File changed:** [`env2_player_bio.py`](../env2_player_bio.py) (+ docs in [`env2_player_bio.md`](../env2_player_bio.md))
**Change:** `near_cr_gain: 1.0 → 2.5` (the gain factor on the NEAR_CR OR gate preactivation)

## Symptom

Running `evaluate(bio_player(), Bot, Environment, seed=12345)`:

| numpy   | mean   | std    | outcome                              |
| ------- | ------ | ------ | ------------------------------------ |
| 1.26.4  | 14.73  | 0.08   | PASS                                 |
| 2.4.4   | varies | large  | run 5/10 (inner seed 86398) crashed  |

Single-run divergence was ~8 distance units on one of ten evaluate runs,
dragging the mean below threshold.

## Root cause

BLAS in numpy 2.x reorders accumulation in `np.dot(W_in, I) + np.dot(W, X)`
relative to numpy 1.26.4. The reorder is ULP-level (~5e-13), but because
the controller is a 1000-neuron recurrent network with sharp
`relu_tanh(k_sharp · …)` gates, the noise amplifies across ~800
iterations.

On seed 86398 (inner seed used in `evaluate(seed=12345)`, run index 5),
after the bot has executed four successful corridor shortcuts, the
`near_e` bump detector lands at:

- numpy 1.26.4: `near_e ≈ 0.522`
- numpy 2.4.4:  `near_e ≈ 0.511`

Both are above the OR gate's 0.5 midpoint, so both `should` trigger
`near_cr`. But the original gate was

```python
near_cr(t+1) = relu_tanh(k_sharp * (near_e + near_w - 0.5))    # k_sharp = 50
```

giving:

- numpy 1.26.4: `relu_tanh(50 · 0.022) = tanh(1.10) = 0.80` → rounds OK
- numpy 2.4.4:  `relu_tanh(50 · 0.011) = tanh(0.55) = 0.50` → borderline

The numpy 2.4.4 value sits *exactly* at the `trig_sc` AND threshold of
3.5 (`reward_latch + heading_horiz + front_clear + near_cr − 3.5 = 0`).
The AND gate then produces `trig_sc ≈ 0.05` at step 810 and
`trig_sc ≈ 0.83` at step 811 — a split firing across two consecutive
steps.

The shortcut countdown reload

```python
sc_countdown(t+1) = max(0, sc_countdown(t) - 1 + (sc_total + 1) * trig_sc(t))
```

with `sc_total = 63` then produces `sc_countdown ≈ 54.3` instead of
`63`, truncating the 63-step shortcut turn by about 9 steps. The bot
exits the corridor at the wrong angle and crashes.

## Fix

Sharpen the NEAR_CR OR gate by a factor of 2.5 so the preactivation
margin at `near_e = 0.511` is large enough that `near_cr` saturates to
~1.0 in a single step:

```python
near_cr_gain = 2.5
W[NEAR_CR, NEAR_E] = near_cr_gain * k_sharp
W[NEAR_CR, NEAR_W] = near_cr_gain * k_sharp
W_in[NEAR_CR, bias_idx] = -0.5 * near_cr_gain * k_sharp
```

At gain 2.5, `relu_tanh(2.5 · 50 · 0.011) = tanh(1.375) = 0.88`, which
gives `trig_sc` a 0.38-unit margin above threshold (`tanh(50 · 0.38) =
1.0`) and fires `trig_sc` cleanly on a single step.

## Safe-gain window

Sweeping `near_cr_gain ∈ {1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7, 10}` on
numpy 2.4.4 across 10 outer seeds:

| gain | seed 12345 (86398) | seed 101 | other 8 seeds |
| ---- | ------------------ | -------- | -------------- |
| 1.0  | FAIL (crash on run 5) | PASS | mixed           |
| 2.0  | PASS (14.737)      | PASS (14.707) | PASS all     |
| 2.5  | PASS (14.737)      | PASS (14.707) | PASS all     |
| 3.0  | PASS (14.737)      | PASS (14.707) | PASS all     |
| 3.5  | PASS (14.737)      | PASS        | — (not exhaustively tested) |
| 4.0  | PASS               | FAIL (14.296) | regressions appear |
| ≥5   | PASS               | FAIL (14.42)  | regressions     |

- **Lower bound (2.0)**: below this, seed 86398's BLAS-drift margin
  vanishes.
- **Upper bound (3.0)**: above this, the sharper gate fires `near_cr`
  one step earlier on the ramp-up of `near_e`, which shifts `trig_sc`
  timing on other outer seeds (seed 101 regresses first at gain 4).
- **Chosen value: 2.5** — midpoint of the safe window `[2.0, 3.0]`.

## Validation

Final 10-seed robustness sweep with `near_cr_gain = 2.5`:

### numpy 2.4.4
```
[PASS] outer=   12345  mean=14.7370  std=0.0814
[PASS] outer=     101  mean=14.7070  std=0.1147
[PASS] outer= 5288742  mean=14.7050  std=0.0850
[PASS] outer=  311895  mean=14.7380  std=0.0613
[PASS] outer=      77  mean=14.6850  std=0.0266
[PASS] outer=     420  mean=14.7160  std=0.1045
[PASS] outer=    2024  mean=14.7670  std=0.0874
[PASS] outer=     999  mean=14.7360  std=0.0888
[PASS] outer=       1  mean=14.6561  std=0.2590
[PASS] outer=  123456  mean=14.7220  std=0.0671
ALL 10 SEEDS PASS.
```

### numpy 1.26.4
```
[PASS] outer=   12345  mean=14.7350  std=0.0818
[PASS] outer=     101  mean=14.7100  std=0.1128
[PASS] outer= 5288742  mean=14.6571  std=0.2044
[PASS] outer=  311895  mean=14.7380  std=0.0613
[PASS] outer=      77  mean=14.6870  std=0.0276
[PASS] outer=     420  mean=14.7160  std=0.1045
[PASS] outer=    2024  mean=14.7680  std=0.0838
[PASS] outer=     999  mean=14.7420  std=0.0889
[PASS] outer=       1  mean=14.7310  std=0.0769
[PASS] outer=  123456  mean=14.7260  std=0.0625
ALL 10 SEEDS PASS.
```

## Things rejected along the way

- **Threshold drop (`trig_sc` AND from 3.5 to 3.2)**: fixed seed 86398
  but broke seed 916765 (premature fire when `front_clear` was
  partially transitioning).
- **`near_cr` threshold from 0.5 to 0.1**: broke 4 of 10 runs on the
  primary seed — earlier transition shifted the trigger `pos_x` outside
  the safe drift-offset window.
- **10× gain (`near_cr_gain = 10`)**: fully fixed seed 86398 but broke
  seed 101 (premature firing). Same root cause as the 4× boundary
  above — just more aggressive.

## Related

- Same sensitivity likely exists in `env2_player_reflex_bio2.py`; try
  the same 2.5× gain on its corridor OR gate as the first thing if that
  file starts failing under numpy 2.x.
- See `plans/reflex_bio2_init_heading_robustness_plan.md` for a similar
  robustness-window investigation (initial-heading seed window width).
