# Reflex Bio Player 2 - Environment 2

## Overview

`env2_player_reflex_bio2.py` is the pointwise-activation version of the env2
bio controller. The network keeps the same evaluation harness and reservoir
size as before (`n = 1000`), but simplifies several helper circuits:

- the trig pair now uses only the `sin` activation,
- `sin_pos/sin_neg` is replaced by `sin_sq`,
- `Xi_hi/Xi_lo` is replaced by a single `xi_blue` bank,
- `x_pos/x_neg/near_center` is replaced by one bump detector,
- `xe/xw` corridor edge pairs are replaced by bump-based `near_e/near_w`,
- dead diagnostics (`X5`, `X12`, `X23` depletion helpers, `tt_*`) are removed.

The accepted implementation was verified locally on April 17, 2026 with the
standard command:

```powershell
python braincraft\env2_player_reflex_bio2.py
```

and matched the existing fixed-harness score:

```text
14.71 +/- 0.00
```

## Activation Library

Every hidden neuron still uses a fixed pointwise scalar activation:

| Activation | Formula | Main use |
| --- | --- | --- |
| `relu_tanh` | `max(0, tanh(z))` | thresholds, latches, logic gates |
| `identity` | `z` | accumulators and passthrough state |
| `relu` | `max(0, z)` | countdowns and counters |
| `clip_a` | `clip(z, -a, a)` | steering output `X6` |
| `sin` | `sin(z)` | both trig neurons |
| `square` | `z^2` | exact sine-magnitude test |
| `bump` | `max(0, 1 - 4z^2)` | compact support detectors |

The new `bump` activation is used in three places:

- `xi_blue[r]` centered at color `4`,
- `near_center` centered at `X10 = 0`,
- `near_e` / `near_w` centered at `X10 = -DRIFT_OFFSET` and `+DRIFT_OFFSET`.

No `xi_red` fallback bank is retained in the accepted design because the
blue-only version already held the baseline score.

## Layout

### Fixed low slots

| Slot | Meaning |
| --- | --- |
| `0..4` | reflex feature neurons |
| `5` | unused legacy slot |
| `6` | clipped steering output |
| `7` | direction accumulator |
| `8, 9` | unused legacy slots |
| `10, 11` | position accumulators |
| `12` | unused legacy slot |
| `13, 14` | initial-correction latch and instantaneous sensor term |
| `15..18` | reward circuit |
| `19` | step counter |
| `20` | `shortcut_steer` |
| `22` | shortcut countdown |
| `23` | `init_impulse` |
| `24` | seeded flag |

### Allocated helper layout (`n_rays = 64`, no red fallback)

| Range | Meaning |
| --- | --- |
| `25..88` | `xi_blue[0..63]` |
| `89..99` | evidence and front-sign helpers |
| `100..106` | trig and sign helpers (`cos_n`, `sin_n`, `sin_sq`, `cos_pos`, `cos_neg`, `y_pos`, `y_neg`) |
| `107..117` | shortcut predicates and phase indicators |
| `118..119` | `seed_pos`, `seed_neg` |
| `120..123` | `cy_pp`, `cy_pn`, `cy_np`, `cy_nn` |
| `124..132` | corridor predicates (`cos_big_*`, `near_e`, `near_w`, `ncr_*`, `cos_small`, `near_corr`) |

The helper names exposed by `_bio2_indices()` now match the accepted design:

- `shortcut_steer`
- `init_impulse`
- `sin_sq`
- `xi_blue_start` / `xi_blue_stop`
- `near_e`
- `near_w`

Deleted names such as `sin_pos`, `x_pos`, `xe_pos`, `tt_plus`, `x23p`, and
`is_init` are no longer part of the allocator.

## Trig and Frame Semantics

The accepted change is trig option **2A**: preserve current outputs while
removing the dedicated `cos` activation branch.

Let:

```text
phi = X7 + X13 + X6
```

Then the two trig neurons use the same `sin` activation with different biases:

```text
COS_N = sin(phi + pi)       = -sin(phi)
SIN_N = sin(phi + pi / 2)   =  cos(phi)
```

This preserves the controller's existing downstream frame exactly. In
particular, the position and corridor logic still uses the same tuned `X10/X11`
semantics as before, so no geometric retune was needed.

## Main Circuits

### Reflex output

`X0..X4` remain the wall-following reflex features. Their output weights are
unchanged, and approach-phase cancellation still suppresses only these five
source neurons.

Legacy `X5` and `X12` are no longer wired. The real front-block signal is
entirely carried by `x5_pos` / `x5_neg`.

### Initial heading correction

The old `X23` depletion loop was removed. The accepted design keeps only:

- `seed_pos` / `seed_neg` to capture the step-0 correction sign,
- `X13` as the long-lived correction latch,
- `init_impulse` at slot `23` as the one-step steering pulse.

`init_impulse` is now just:

```text
init_impulse = -seed_pos + seed_neg
```

and feeds the output directly with weight `1.0`.

### Color evidence

Each ray uses one bump detector centered on blue:

```text
xi_blue[r] = bump(color[r] - 4)
```

The left and right evidence neurons are summed blue counts:

```text
L_EV = sum(xi_blue[left half])
R_EV = sum(xi_blue[right half])
```

The red-suppression fallback was tested as optional only; it was not needed for
the accepted controller.

### Heading magnitude tests

Instead of the old paired rectifiers:

```text
sin_sq = SIN_N^2
```

and the two heading predicates are exact threshold tests on `sin^2`:

```text
heading_vert  : sin_sq > 0.70^2
heading_horiz : sin_sq < 0.35^2
```

### Center and corridor predicates

`near_center`, `near_e`, and `near_w` are bump detectors:

```text
near_center = bump(X10 / (2 * NEAR_CENTER_THR))
near_e      = bump((X10 + DRIFT_OFFSET) / (2 * NEAR_CENTER_THR))
near_w      = bump((X10 - DRIFT_OFFSET) / (2 * NEAR_CENTER_THR))
```

With `NEAR_CENTER_THR = 0.05`, the support is still `|value| < 0.05`.

Because bump detectors taper at the band edge instead of switching sharply, the
AND-style gates that consume them use a slightly lower bias (`1.2` instead of
`1.5`) for:

- `nc_and_hv`
- `ncr_e`
- `ncr_w`
- `ncr_c`

### Shortcut outputs

The old combined `X20` output is now split into two fixed-slot channels:

```text
shortcut_steer = |SHORTCUT_TURN| * (cy_pn + cy_np - cy_pp - cy_nn)
init_impulse   = -seed_pos + seed_neg
```

Both feed `Wout` with weight `1.0`, so the final steering sum is unchanged.

## Verification

The accepted controller passed all of the following after the rewrite:

```powershell
python braincraft\_debug_bio2.py --steps 120 --stride 20
python braincraft\_debug_bio2_detail.py --steps 120
python braincraft\env2_player_reflex_bio2.py
```

Observed outcome:

- the side-by-side trace stayed aligned with the previous controller,
- the detail debug script handled optional neuron names correctly,
- the final env2 score remained `14.71 +/- 0.00`.
