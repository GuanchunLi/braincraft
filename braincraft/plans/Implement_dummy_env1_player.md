# Plan: Dummy Env1 Player — Outer-Ring Clockwise Walker

## Goal
Build a hand-crafted ESN-form player for Task 1 (`environment_1`) that:
1. During the initial steps, drives the bot straight up from the start position (0.5, 0.5, heading 90°) until it reaches the outer wall.
2. Then sticks to the outer ring and circulates **clockwise** (top → right → bottom → left → top).

The player must satisfy the BrainCraft constraints: the only learned/returned artifacts are `(W_in, W, W_out, warmup, leak, f, g)` consumed by the ESN equations in [challenge_1.py:145-162](braincraft/challenge_1.py#L145-L162). No state outside `X` may be kept between steps.

Output is the file [braincraft/env1_player_dummy.py](braincraft/env1_player_dummy.py) (new), structured like [braincraft/env1_player_random.py](braincraft/env1_player_random.py).

## Key facts gathered from the code
- Input vector `I` has length `n+3 = 67`: `I[:64] = 1 - camera.depths` (closeness, larger = closer wall), `I[64] = bot.hit`, `I[65] = bot.energy`, `I[66] = 1.0` (bias). See [challenge_1.py:150-153](braincraft/challenge_1.py#L150-L153).
- `n = 1000` reservoir neurons. Update: `X = (1-λ)X + λ f(W_in I + W X)`, output `O = W_out g(X)`.
- `bot.forward` interprets `O` as **radians** and clamps to `[-π/36, +π/36]` (i.e. ±5°). See [bot.py:138](braincraft/bot.py#L138). So target output magnitude for "max right turn" ≈ `-0.0873`.
- Camera FOV is 60° spread over 64 sensors → roughly 1° per sensor. Index 0 is leftmost (≈+30°), index 63 is rightmost (≈-30°); "front" is around index 31-32. (Verify when implementing by inspecting [camera.py](braincraft/camera.py); if convention is reversed, swap left/right masks — the design is symmetric in concept.)
- Initial heading is 90° ± 5° (random noise, [bot.py:30](braincraft/bot.py#L30)). The bot must therefore tolerate small initial angle noise.
- `bot.energy` starts at 1.0, decays each step; cannot rely on it as a phase indicator beyond rough thresholds.

## Strategy: minimal hand-designed ESN

We use the 1000-neuron reservoir as a thin functional layer — most weights are zero. Conceptually we only need ~3 "feature" neurons; padding the remainder with zeros keeps the contract intact.

### Behavior specification (control law)
Let `c[i] = 1 - depths[i]` ∈ [0,1] (closeness).
Define three pooled features over the 64 distance sensors:
- `front = mean(c[28:36])` — closeness directly ahead (8 central rays).
- `left  = mean(c[0:16])` — closeness on the left side.
- `right = mean(c[48:64])` — closeness on the right side.

Desired steering (in radians, before clipping):
```
O = -K_front * max(0, front - τ_front)        # turn right hard if wall ahead
    + K_align * (right - left)                # nudge: if right side closer, steer left, vice versa
    - K_bias                                    # tiny constant rightward bias
```
- The first term causes a right turn whenever a wall is in front → handles every outer-ring corner and the initial collision with the top wall.
- The second term keeps the bot parallel to the outer wall on its **left** (clockwise traversal puts the outer wall on the left side, so we want `left` to stay larger than `right`; if `left < right`, we steer right, otherwise slightly left). NOTE: if camera index ordering turns out to be the opposite (index 0 = right), simply swap the `left`/`right` slice definitions — the rest of the design is unchanged.
- The constant bias guarantees the bot eventually meets a wall during the warmup phase even if it starts drifted slightly off-90°.

Tuning starting points (to refine empirically): `K_front = 0.4`, `τ_front = 0.5`, `K_align = 0.05`, `K_bias = 0.005`.

### Mapping the control law into the ESN form
We need `O = W_out · g(X)` after one update of `X`. We pick `g = identity`, `f = identity` (so the reservoir is linear and computes pooled sums exactly), `λ = 1.0` (no leak — `X(t+1)` depends only on the current input, not on history). With `W = 0` (zero recurrent matrix) and `λ = 1`, equation 1 reduces to `X = W_in · I`.

Then the output is `O = W_out · W_in · I`, which is a fixed linear function of the current sensor vector — exactly what our control law needs.

Allocate three "feature neurons" inside the 1000-neuron reservoir (e.g. indices 0, 1, 2):
- Neuron 0 ← `front` pooled closeness:
  `W_in[0, 28:36] = 1/8`, all other entries in row 0 are 0.
  But we actually want `max(0, front - τ_front)`. With `f = identity` we cannot apply a ReLU. Two options:
  1. **Switch `f` to `np.tanh`** and bias the neuron strongly so its operating point sits on the steep part: set `W_in[0, 28:36] = a`, `W_in[0, 66] = -a*τ_front` (uses the constant-1 bias input). After `tanh`, the neuron approximates a soft-thresholded "wall ahead" indicator. This is the cleanest fit.
  2. Keep `f = identity` and let `W_out` apply a negative weight. The result is a linear (not thresholded) front-avoidance term, which still works but is gentler near corners.
  Decision: use option 1 (`f = np.tanh`, `g = identity`, `λ = 1.0`). Tanh is the standard ESN activation and matches what other players use.
- Neuron 1 ← `left - right` (signed alignment): `W_in[1, 0:16] = +1/16`, `W_in[1, 48:64] = -1/16`. (Or swap if camera ordering is reversed.) After tanh this is monotonic in the signed difference, near-linear for small differences.
- Neuron 2 ← constant 1 from the bias input: `W_in[2, 66] = arctanh(0.5) ≈ 0.549` so that `tanh(...) ≈ 0.5`, giving a stable nonzero "constant" feature. Or simply set `W_in[2, 66] = 2.0` and accept `tanh(2)≈0.964`; the exact value is absorbed by `W_out`.
- All other 997 rows of `W_in` are zero.

`W` is the zero matrix (1000×1000).

`W_out` is shape `(1, 1000)`, all zeros except:
- `W_out[0, 0] = -K_front_eff` (turn right when front neuron fires; sign chosen so positive front activation produces negative O = right turn).
- `W_out[0, 1] = -K_align_eff` (if `left - right > 0`, we want a small right turn? No — if the left wall is too close we need to steer right, so the sign here depends on the convention. We will set the sign so that the bot tracks the wall on its left at a target distance: if left > right (left wall closer) → steer right → `O < 0`. So coefficient on neuron 1 is **negative**.)
- `W_out[0, 2] = -K_bias_eff` (constant rightward nudge so the bot starts curving in the warmup-free phase).

Concrete starting magnitudes (will be tuned by a tiny grid search during training): `K_front_eff = 0.09`, `K_align_eff = 0.04`, `K_bias_eff = 0.002`. These are radians; total |O| stays well within the ±0.0873 clip when only one term saturates.

`warmup = 0`. The bot starts facing up; it will simply walk forward until the front sensors trip the corner-turn term. No need for an explicit "go straight" phase.

`leak = 1.0`, `f = np.tanh`, `g = identity`.

## Training function
Following the structure of [env1_player_random.py](braincraft/env1_player_random.py):

1. Build the hand-crafted `(W_in, W, W_out, warmup, leak, f, g)` model from the constants above. `yield` it once immediately so there is always a valid result even if training is killed.
2. Run a small grid search over `(K_front_eff, K_align_eff, K_bias_eff, τ_front)` — say 3 values per knob, 81 combinations max — calling `evaluate(model, Bot, Environment, runs=3, debug=False)`. Keep the best by mean score (tie-break by lower std).
3. After each candidate, `yield` the current best so the harness always has a returnable model under the 100 s wall clock.
4. The evaluator decides the order of the camera indices, so the first iteration of the grid search should also try the **left/right swapped** variant of `W_in` — encode this as one extra binary knob `swap_lr ∈ {0,1}`.
5. Set `np.random.seed(1)` at the top of the player (mirroring `env1_player_random.py`) so the constants are deterministic even though the design is not random.

## `__main__` block
Identical pattern to [env1_player_random.py](braincraft/env1_player_random.py#L49-L66): import `train, evaluate` from `challenge_1`, set `seed = 12345`, call `train(dummy_player, timeout=100)`, then `evaluate(model, Bot, Environment, debug=False, seed=seed)`, print mean ± std.

## Validation steps (manual, after implementation)
1. Run `python braincraft/env1_player_dummy.py` from the repo root. It must finish within ~5 minutes wall time and print a score.
2. Sanity-check by running a single eval with `debug=True` (temporarily) and visually verifying that the bot:
   - Starts in the centre, walks up, hits the top wall, turns right.
   - Then circulates clockwise along the outer perimeter, never penetrating the central columns.
3. Expected score range based on the README leaderboard for ring-walking strategies on Task 1: somewhere in the **8–14** band (the obstacles in env1 — the two central columns — make the outer ring slightly less efficient than in env3 where wall-following scores ~15). Anything below ~6 indicates a sign error (probably the left/right swap) or that the front threshold is wrong.

## Files touched
- **Add** [braincraft/env1_player_dummy.py](braincraft/env1_player_dummy.py) — the only new file.
- No changes to existing files. README leaderboard update is **not** part of this task.

## Out of scope (do not do without explicit approval)
- Implementing the player code itself.
- Adding the player to the README leaderboard.
- Any modification of the `bot`, `camera`, `environment_1`, or `challenge_1` modules.
- Learning algorithms beyond the small hand-tuning grid search described above.
