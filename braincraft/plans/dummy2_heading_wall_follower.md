# dummy2: heading-aligned wall follower for env1

## Context

The current [env1_player_dummy.py](braincraft/env1_player_dummy.py) is a CW outer-ring follower whose steering during the wall-following phase is dominated by a single side-proximity ray (idx 11). The user wants a second variant ("dummy2") whose trajectory is essentially the same (CW outer ring) but whose steering during the *following* phase is driven by head-direction-like information rather than side proximity. Side proximity stays only as a safety valve. The corner-turn behavior (hard turn on `hit`) and the very first wall approach should remain hit-driven.

The trigger observation: when the existing dummy is stably aligned with a wall its heading sits near a multiple of 90°, and the network output drops near zero at those moments. The new controller should make this explicit — steer to zero out (heading mod 90°).

## Key constraint

The bot exposes no direct heading sensor. Inputs are 64 camera depths + `hit` + `energy` + bias (see [challenge_1.py:150-152](braincraft/challenge_1.py#L150-L152)). However, the env1 world ([environment_1.py:48-59](braincraft/environment_1.py#L48-L59)) is fully axis-aligned, so heading-aligned-to-axis is observable from **left/right symmetry of the depth profile**: when heading is a multiple of 90°, a pair of rays placed symmetrically around the camera center see structurally symmetric depths; any tilt off-axis breaks that symmetry monotonically. This gives a linear, network-friendly proxy for `(heading mod 90°)` that is "internal" in the sense the user means — it reflects the bot's orientation, not its distance to a particular wall.

## Approach

Create [braincraft/env1_player_dummy2.py](braincraft/env1_player_dummy2.py) by copying [env1_player_dummy.py](braincraft/env1_player_dummy.py) and replacing the wall-following feature mix. Same ESN scaffolding (`n=1000`, identity activations, `leak=1`, single yield), so it plugs into `train` / `evaluate` from [challenge_1.py](braincraft/challenge_1.py) unchanged.

### Feature neurons (X0..X4)

- `X0 = hit`  → corner / first-wall hard turn (unchanged from dummy).
- `X1 = bias` → constant 1.
- `X2 = prox[L]` with `L = 20` (left of center, ~+9° off heading).
- `X3 = prox[R]` with `R = 43` (right of center, symmetric around the 31/32 midline, ~−9° off heading). `R = 63 - L`.
- `X4 = prox[side_idx]` with `side_idx = 11` (the existing left-side ray) — used **only** as a safety valve.

`prox[i] = 1 - depth[i]` is already what the harness writes into `I[:n,0]` ([challenge_1.py:150](braincraft/challenge_1.py#L150)), so `Win[k, i] = 1.0` is enough to copy a ray into a feature neuron, exactly as the existing dummy does for `side_idx`.

### Output mix (Wout, single readout)

```
O = hit_turn * X0
  + heading_gain * (X2 - X3)        # main following signal: zero when heading ⟂ multiples of 90°
  + safety_gain  * relu(X4 - safety_threshold)   # only kicks in when left side is dangerously close
```

- `hit_turn ≈ -5.0` (same as dummy; hard CW snap on contact and on first wall hit).
- `heading_gain`: sign chosen so that when the bot drifts CCW off-axis (left ray sees more open space than right ray during CW following), output is negative (steer right, back onto axis). Start near `-2.0`, tune in eval.
- The safety valve cannot be a true ReLU in a linear-identity ESN with one readout. Two options, in order of preference:
  1. **Linear approximation, no nonlinearity:** drop the ReLU and use a small `safety_gain` together with a bias offset, so `safety_gain * (X4 - safety_threshold)` is always present but small enough to be dominated by the heading term unless `X4` spikes. This preserves the "safety valve, not mainstream" property as long as `|safety_gain| ≪ |heading_gain|` and `safety_threshold` is set near the normal cruising proximity (~0.65, matching the existing dummy's `wall_target`).
  2. If option 1 underperforms in eval, switch the activation `f` from `identity` to `relu_tanh` (already defined in [env1_player_dummy.py:24-28](braincraft/env1_player_dummy.py#L24-L28)) for the safety neuron only — but this changes all neurons under the current scaffolding, so prefer option 1 first.

### What is *not* changing

- `n`, `warmup`, `leak`, activations, `W` (zero), single `yield model`, `__main__` block — copy verbatim from [env1_player_dummy.py](braincraft/env1_player_dummy.py) so behavior under `train`/`evaluate` is identical to dummy except for the readout.
- The CW direction and outer-ring trajectory: identical, because `hit_turn` sign and magnitude are unchanged and the heading term, by construction, only nudges the bot back onto an axis-aligned heading — it does not pick a direction of travel.

## Files

- **New:** [braincraft/env1_player_dummy2.py](braincraft/env1_player_dummy2.py)
- **Reference (read-only):** [braincraft/env1_player_dummy.py](braincraft/env1_player_dummy.py), [braincraft/challenge_1.py](braincraft/challenge_1.py), [braincraft/environment_1.py](braincraft/environment_1.py), [braincraft/bot.py](braincraft/bot.py)

## Verification

1. `python braincraft/env1_player_dummy2.py` — should train (instant, single yield) and print a final score ± std over 10 runs.
2. Compare score to the dummy baseline; dummy2 should be in the same ballpark (similar trajectory).
3. Visual check: run with `evaluate(..., debug=True, seed=12345)` in a scratch script and confirm (a) CW outer-ring trajectory, (b) corner turns still happen on `hit`, (c) during straight wall-following segments steering visibly tracks left/right depth symmetry rather than the single side ray.
4. Optionally, add a dummy2 series to [braincraft/env1_trajectory_plot.py](braincraft/env1_trajectory_plot.py) to overlay against the original dummy.

## Open question for user

Tuning of `heading_gain`, `safety_gain`, `safety_threshold`, and the choice of symmetric ray indices (`L=20, R=43` vs. wider/narrower) is best done empirically once the file exists. Plan assumes the user is OK with picking initial values and iterating in a follow-up.
