# Plan for improving `reflex_bio2` player in env2

Scope: simplify and tighten the bio2 pointwise network in
[env2_player_reflex_bio2.py](../env2_player_reflex_bio2.py) without regressing
its evaluation score. All changes are bio2-only (bio1 untouched) so that
`_debug_bio2.py` still provides a side-by-side baseline.

All edits are concentrated in three places:
- [`_bio2_indices()`](../env2_player_reflex_bio2.py#L97) — allocation table
- [`make_activation()`](../env2_player_reflex_bio2.py#L195) — pointwise f
- [`reflex_bio2_player()`](../env2_player_reflex_bio2.py#L230) — wiring

Refactor order is chosen so each step is individually testable and reversible.

---

## Validation protocol (run after every step)

Each step should leave the player score unchanged (or changed by a known,
justified amount). Baseline first, then run after each step:

1. **Step-trace equivalence** (cheap, catches almost everything):
   ```
   python _debug_bio2.py --steps 200 --stride 10
   ```
   Compare the BIO2 column against its previous baseline. `O`, `pos`, `E`
   at each printed row should match to ≥ 4 decimals for steps where the
   refactor is mathematically equivalent (steps 1, 2, 3, 5, 7, 8). Steps 4
   and 6 may shift values slightly (new activation/approximation) — record
   the delta and keep it justified.

2. **Detail trace at key steps** (when a value drift is suspected):
   ```
   python _debug_bio2_detail.py   # adjust step list inside
   ```

3. **Full score** (final gate before merging a step):
   ```
   python challenge_2.py          # evaluate(..., runs=10)
   ```
   Current bio2 baseline ≈ 14.71 (from
   [env2_player_reflex_bio2.md:30](../env2_player_reflex_bio2.md#L30)).
   Record mean ± std after each step.

If a step drops score by more than ~0.05, revert and re-examine before
proceeding.

---

## Decisions (locked after review round 1)

1. **Item 6 scope** = *per-pair audit*: collapse a pair to a single
   squared-activation neuron **iff every downstream consumer uses only
   the sum** of the pair. Sign-preserving pairs stay as is. See the pair
   audit table in Step 6 for the verdict on each of the 7 candidate pairs.
2. **Item 4 / bump activation** = add one new pointwise activation class
   `bump(x) = max(0, 1 − 4·x²)` with support `x ∈ [−0.5, +0.5]`, peak 1
   at `x=0`, zero outside. The narrower support gives a crisper detector
   than `1 − x²` and composes with all re-usable magnitude collapses in
   Step 6 (any `|value| < THR` test scales as `bump(value / (2·THR))`).
3. **Item 2 / cos bug** — needs one more decision (Step 2A vs 2B below);
   `−π/2` as proposed does not work mathematically. See Step 2 for the
   derivation and two concrete options.
4. **Item 5 / X19 gain** = `−500` (replacing `−1.0e6`). Headroom ~8× the
   worst-case pre-reset counter value.
5. **Item 8 / X20 split names** = `shortcut_steer` and `init_impulse`
   (no `20` in either name, semantically distinct, self-documenting).

---

## Step 0 — Baseline snapshot

Before any edit, capture the current behavior so every later step has
something to diff against.

1. Run `python _debug_bio2.py --steps 400 --stride 20 > bio2_baseline.txt`
2. Run `python challenge_2.py` and record mean/std.
3. Commit these artifacts (or paste into PR description) — not the code.

No code changes in this step.

---

## Step 1 — Remove dead neurons X5, X8, X9, X12

**Goal:** delete neurons with no downstream consumer. Pure cleanup, zero
behavior change expected.

Investigation summary (from
[env2_player_reflex_bio2.py:310–368](../env2_player_reflex_bio2.py#L310)):

| Neuron | State | Notes |
|---|---|---|
| X5 | Diagnostic only | `Wout[0,5]=0`; not read by W. X5P/X5N computed independently. |
| X8, X9 | Placeholder comments only | Real trig is `COS_N`/`SIN_N`. No `Win`/`W` entries. |
| X12 | Diagnostic only | `Win[12, hit_idx]=1.0` set but never read. |

### Edits

1. In [`reflex_bio2_player()`](../env2_player_reflex_bio2.py#L230):
   - Delete lines setting `Win[5, C1_idx]`, `Win[5, C2_idx]`, `Win[5, bias_idx]`
     (currently [330–332](../env2_player_reflex_bio2.py#L330)) and the
     constants `C1_idx, C2_idx, front_thr` if unused elsewhere (verify).
   - Delete `Wout[0, 5] = 0.0` (line 349 — redundant but kept for clarity).
   - Delete `Win[12, hit_idx] = 1.0` and the "X12" comment block
     ([367–368](../env2_player_reflex_bio2.py#L367)).
   - Remove references to "X8, X9" from the docstring at
     [line 39](../env2_player_reflex_bio2.py#L39).
2. The approach-phase silence loop at
   [line 325](../env2_player_reflex_bio2.py#L325) (`for i in range(5)`)
   currently silences X0–X4 using IS_APP. X5 is not in this loop, so no
   change needed. Keep the loop at `range(5)`.
3. Nothing to remove in `_bio2_indices()` — X5, X8, X9, X12 were never
   allocated by name; they are just fixed low indices.

### Risks & verification

- `Wout[0,5]=0` has always been 0, so removing the explicit assignment is
  safe but produces no score change.
- Run Step-trace equivalence; expect **bit-identical** output at every row.
- Keep `n = 1000` (neuron array size) unchanged — slots 5, 8, 9, 12 simply
  remain zero. Renumbering is out of scope here.

---

## Step 2 — Unify cos / sin activation (and decide whether to fix the naming bug)

**Goal:** remove the `cos` branch from `make_activation()`; use only
`sin` for both trig neurons. Requires one decision first (2A vs 2B).

### The current situation (verified from code)

[env2_player_reflex_bio2.py:536–540](../env2_player_reflex_bio2.py#L536)
wires **identical** pre-activations into both neurons:

```python
for idx_trig in (COS_N, SIN_N):
    W[idx_trig, 7]   = 1.0        # X7 (direction integral)
    W[idx_trig, 13]  = 1.0        # X13 (locked init correction)
    W[idx_trig, 6]   = 1.0        # X6 (current dtheta)
    Win[idx_trig, bias_idx] = np.pi / 2
```

Let `θ ≈ X7 + X13 + X6` (direction angle). Then:

| Neuron | Activation | Output = | Actually is |
|---|---|---|---|
| COS_N | `np.cos` | `cos(θ + π/2)` | **`−sin(θ)`** |
| SIN_N | `np.sin` | `sin(θ + π/2)` | **`cos(θ)`** |

So COS_N outputs `−sin(θ)` and SIN_N outputs `cos(θ)` — the names and
outputs don't match. This is the "hidden bug" I flagged. Downstream
integration is consistent with this (bugged) naming:

- `W[10, COS_N] = speed` → X10 integrates `−speed·sin(θ)` ⇒ X10 = −y
  (negated y-coordinate)
- `W[11, SIN_N] = speed` → X11 integrates `+speed·cos(θ)` ⇒ X11 = +x

So X10/X11 are effectively (−y, +x), i.e. the coordinate frame is
rotated 90° from natural (x, y). All threshold tests on X10/X11
(`NEAR_CENTER_THR`, `DRIFT_OFFSET`, `Y_POS`/`Y_NEG` → cy_*) have been
tuned against this rotated frame.

### Why `−π/2` (as proposed) does not work

Checking `sin(θ + δ) = ?` for candidate offsets:

| δ | sin(θ+δ) | Equals current COS_N (`−sin θ`)? |
|---|---|---|
| `−π/2` | `−cos(θ)` | No (wrong function) |
| `0`    | `sin(θ)`  | No (wrong sign) |
| `+π/2` | `cos(θ)`  | No (wrong sign and function) |
| `+π`   | `−sin(θ)` | **Yes** |
| `−π`   | `−sin(θ)` | **Yes** |

So if we insist on preserving current COS_N output with activation=sin,
the bias must become **`+π`** (or `−π`), not `−π/2`.

### Two options — please choose

**Option 2A — preserve behavior, unify activation only.** Minimal.

- `Win[SIN_N, bias_idx] = π/2` (unchanged)
- `Win[COS_N, bias_idx] = π` (changed from `π/2`)
- Both neurons use `sin` activation → COS_N output `= sin(θ+π) = −sin(θ)`
  (same as current), SIN_N output `= sin(θ+π/2) = cos(θ)` (same as
  current)
- No downstream changes. Step-trace bit-identical.
- Keeps the naming bug. `COS_N` still outputs `−sin(θ)`, `SIN_N` still
  outputs `cos(θ)`. But the only practical effect of the bug is
  confusion when reading code; physical behavior is unchanged.

**Option 2B — fix the bug, audit downstream.** Semantic fix.

- `Win[SIN_N, bias_idx] = 0` → SIN_N output `= sin(θ)` (true sin) ✓
- `Win[COS_N, bias_idx] = π/2` → COS_N output `= sin(θ+π/2) = cos(θ)` ✓
- Both neurons use `sin` activation. Activation case deduplicated.
- Every downstream consumer of COS_N / SIN_N / X10 / X11 needs audit
  because output values change. Enumerated below.
- Step-trace will NOT match baseline. Score may drift in either
  direction. Requires careful re-calibration.

**Downstream consumers to audit under 2B** (from my code read):

| Site | Line | Current meaning | After 2B |
|---|---|---|---|
| `W[10, COS_N]=speed` | 363 | X10 = −y (rotated) | X10 = +x (natural) |
| `W[11, SIN_N]=speed` | 365 | X11 = +x (rotated) | X11 = +y (natural) |
| `W[X_POS, 10]=K_POS` (and NEG) | 561 | thresholds on −y | thresholds on +x |
| `W[Y_POS, 11]=K_SHARP` (and NEG) | 555 | sign of +x | sign of +y |
| `W[XE_P, 10]`, `W[XE_N, 10]` | 618 | corridor offset on −y | corridor offset on +x |
| `W[XW_P, 10]`, `W[XW_N, 10]` | 625 | corridor offset on −y | corridor offset on +x |
| `W[COS_POS, COS_N]=K_SHARP` | 551 | sign of `−sin(θ)` | sign of `cos(θ)` |
| `W[COS_NEG, COS_N]=−K_SHARP` | 552 | sign of `sin(θ)` | sign of `−cos(θ)` |
| `W[COSBP, COS_N]=K_SHARP` | 610 | `−sin(θ) > 0.5` test | `cos(θ) > 0.5` test |
| `W[COSBN, COS_N]=−K_SHARP` | 613 | `−sin(θ) < −0.5` test | `cos(θ) < −0.5` test |

The cy_pp/cy_pn/cy_np/cy_nn quadrant neurons pair `cos_pos/cos_neg`
with `y_pos/y_neg`. Under 2B their semantics change: e.g. `cy_pp =
cos_pos AND y_pos` currently means (`−sin θ > 0`) AND (`+x > 0`); after
2B it means (`cos θ > 0`) AND (`+y > 0`). These are different physical
conditions and will cause the shortcut-steering quadrant labels to
rotate 90°.

The cleanest way to do 2B is: after changing the trig offsets, *also
swap the X10/X11 roles and rename* so the code reads naturally; then
re-check with `_debug_bio2_detail.py` that the steering output at each
step is numerically equivalent to 2A. This is a sizeable audit but it
removes a real code-reading hazard.

### Recommendation

- If you want only a cleanup pass: **go 2A** (one-line change, bit-identical).
- If you want to eliminate the naming hazard: **go 2B** (audit the 10
  sites above, add a doc note to the `.md` file about the new frame).

### Verification (either option)

- After Step 2 runs: `python _debug_bio2.py --steps 400 --stride 20`
- 2A: expect ≥ 1e-9 agreement with baseline at every row.
- 2B: expect divergence from step 0; what matters is the *final score*
  from `challenge_2.py` — record mean ± std.

---

## Step 3 — Remove dead-loop quartet X23 / X23P / X23N / IS_INIT

**Goal:** delete four neurons that form an isolated self-consuming loop
with no downstream effect.

Current behavior
([lines 437–444, 727–739](../env2_player_reflex_bio2.py#L437)):
- `X23` integrates SEEDP − SEEDN (captured at step 0), then depletes via
  X23P/X23N.
- `X23P`, `X23N` are sign thresholds on X23.
- `IS_INIT` reads X23P, X23N, and X24 but is never consumed (not in Wout,
  not in W).
- `SEEDP/SEEDN` already feed `X20` directly at
  [lines 428–429](../env2_player_reflex_bio2.py#L428), so the one-shot
  init correction does **not** depend on X23.

### Edits

1. In `_bio2_indices()`:
   - Delete `alloc("x23p")`, `alloc("x23n")`, `alloc("is_init")`.
   - (Keep `seed_pos`, `seed_neg` — they are live.)
2. In `reflex_bio2_player()`:
   - Remove index aliases `X23P`, `X23N`, `ISI` at
     [lines 289–291](../env2_player_reflex_bio2.py#L289).
   - Remove the `# ── X23` block
     [lines 437–444](../env2_player_reflex_bio2.py#L437).
   - Remove the depletion-branch block
     [lines 727–732](../env2_player_reflex_bio2.py#L727).
   - Remove the IS_INIT block
     [lines 734–739](../env2_player_reflex_bio2.py#L734).
3. In `make_activation()`:
   - Remove `23` from `id_list`
     ([line 201](../env2_player_reflex_bio2.py#L201)).
4. Update the docstring / layout comments that mention X23, X23p, X23n,
   is_init.

### Verification

- Step-trace must be bit-identical — these neurons produce no output.
- Confirm `grep -nE "(X23|x23|is_init|ISI)"` shows zero hits after the
  edit.

---

## Step 4 — Replace Xi_hi / Xi_lo pair with single Xi_blue (bump detector)

**Goal:** cut 128 color-detector neurons to 64 while preserving
“blue ≈ 4” selectivity per ray.

Uses the new `bump(x) = max(0, 1 − 4·x²)` activation (support
`x ∈ [−0.5, +0.5]`, peak 1 at 0).

### Current behavior
([lines 466–481](../env2_player_reflex_bio2.py#L466)):
```python
Xi_hi[r] = relu_tanh(color[r] − 3.5)           # fires when color ≥ 3.5
Xi_lo[r] = relu_tanh(color[r] − 4.5)           # fires when color ≥ 4.5
L_EV = Σ_{r<32} ( Xi_hi[r] − 3·Xi_lo[r] )
R_EV = Σ_{r≥32} ( Xi_hi[r] − 3·Xi_lo[r] )
```
Per-ray response (after relu_tanh saturates for large z):

| `color[r]` | 0 (green) | 3 | 3.5 | 4 (blue) | 4.5 | 5 (red) |
|---|---|---|---|---|---|---|
| `Xi_hi − 3·Xi_lo` | 0 | 0 | ~0 | ~1 | ~0 | **~−2** |

So the current aggregator gives **positive** signal for blue, zero for
green, and **strongly negative** signal for red. That asymmetric
"red-is-bad" signal is useful: it not only rewards blue rays but
penalizes red rays on the same side.

### Proposed rewrite

Per-ray: `z = color[r] − 4`, activation `bump(z) = max(0, 1 − 4z²)`,
support `color[r] ∈ [3.5, 4.5]`:

| `color[r]` | 0 | 3 | 3.5 | 4 | 4.5 | 5 |
|---|---|---|---|---|---|---|
| `bump(color − 4)` | 0 | 0 | 0 | 1 | 0 | 0 |

**Behavior change**: red rays no longer contribute a negative signal.
`L_EV` / `R_EV` become pure blue-counters. The `COLOR_EVIDENCE_THRESHOLD
= 2.0` trigger now requires 2+ blue rays on one side vs the other,
rather than being boosted by contralateral red rays.

Whether this hurts the score depends on how often the old red-penalty
actually fires the TRIG_POS/TRIG_NEG trigger earlier than the pure
blue-count would. Expected impact: small (the trigger fires mostly on
blue-rich scenes anyway), but needs empirical check.

### Optional fallback if score regresses

If removing the red penalty regresses score noticeably, add a second
bump neuron per ray at `color = 5`:

```python
# Xi_red[r]: bump(color[r] − 5), support [4.5, 5.5]
W[L_EV, Xi_red[r]] = −3.0            # mimic old "−3·Xi_lo" red penalty
```

This still uses 128 neurons but they are now two physically meaningful
bumps (blue detector + red detector) instead of two monotone thresholds.
We can defer this decision to after the initial score measurement.

### Edits

1. In `_bio2_indices()` [line 99](../env2_player_reflex_bio2.py#L99):
   - Replace `xi_hi_start/xi_hi_stop/xi_lo_start/xi_lo_stop` with
     `xi_blue_start = 25`, `xi_blue_stop = 25 + n_rays`.
   - All subsequent `alloc(...)` slots shift by `−n_rays` (−64).
2. In `make_activation()` [line 195](../env2_player_reflex_bio2.py#L195):
   - Add `bump_arr` from `idx["xi_blue_start"] : idx["xi_blue_stop"]`.
   - Add branch:
     `out[bump_arr, 0] = np.maximum(0.0, 1.0 − 4.0 * x[bump_arr, 0]**2)`.
3. In `reflex_bio2_player()` [lines 466–481](../env2_player_reflex_bio2.py#L466):
   - Replace the hi/lo loop with a single Xi_blue loop:
     ```python
     for r in range(n_rays):
         Win[idx["xi_blue_start"] + r, p + r]    = 1.0
         Win[idx["xi_blue_start"] + r, bias_idx] = -4.0
     for r in range(half):
         W[L_EV, idx["xi_blue_start"] + r] = 1.0
     for r in range(half, n_rays):
         W[R_EV, idx["xi_blue_start"] + r] = 1.0
     ```
   - Delete `BLUE_HI_THR`, `BLUE_LO_THR`, `XI_LO_FACTOR` constants
     ([lines 86–88](../env2_player_reflex_bio2.py#L86)).
4. Update neuron layout comment at the top of `_bio2_indices()`.

### Risks & verification

- Step-trace will diverge on steps where the old red-penalty would have
  tipped an evidence trigger. Record per-step differences.
- Run full `challenge_2.py`. If score drops > 0.1, enable the
  `Xi_red` fallback above before moving on.

---

## Step 5 — Reduce the X19 reset gain to 500

**Goal:** replace the cosmetically-huge `−1e6` NCV coupling with 500
(locked per decision 4).

Current
([line 413](../env2_player_reflex_bio2.py#L413)):
```python
W[19, NCV] = -float(1.0e6)
```
`X19` is a `relu` counter incremented by `+1`/step, typically reset
around `COUNTER_THR = 60`. With `NCV ≈ 1` at reset, `z[19] = X19_prev +
1 − 500`, which goes negative for any `X19_prev < 499`. Headroom is ~8×
the expected max — plenty.

### Edits

1. Change the constant on
   [line 413](../env2_player_reflex_bio2.py#L413) to `-500.0`.
2. Replace the `# huge negative pulse resets to 0` comment with one line
   noting the ~8× headroom above COUNTER_THR.

### Safety bound (why 500 is safe)

Worst case for `X19_prev`: if NCV fails to fire for an extended period,
X19 accumulates at +1/step. Run count is 100 s × some step rate — but
even at 1 kHz and no reset ever, X19 ≤ 1e5 still satisfies `X19 + 1 −
500·NCV ≤ 0` whenever NCV ≈ 1. So `G = 500` is more than enough under
normal operation; only a pathological case with X19 > 500 would
under-reset, and that case would already be catastrophic.

### Verification

- Step-trace bit-identical. X19 reset is binary; any `G ≥ max(X19) + 1`
  gives the same behavior. Expect ≥ 1e-9 agreement with baseline.
- If step-trace diverges: X19 has accumulated above 500 at some point.
  Raise G (e.g. to 2000) and investigate why NCV isn't firing.

---

## Step 6 — Collapse magnitude-only `_pos/_neg` pairs via `x²` / bump

**Goal:** per your instruction, audit every `_pos/_neg` pair and
collapse it to a single neuron *only when the sum is the only thing
downstream ever reads*. For sign-using pairs, keep the split.

### Pair-by-pair audit (result of code walk)

| Pair | Downstream consumers | Sum-only? | Verdict |
|---|---|---|---|
| `SIN_POS` / `SIN_NEG` | `HV`, `HH` (both use `SIN_POS + SIN_NEG`) | **yes** | **Collapse** → `sin_sq` (x²) |
| `COS_POS` / `COS_NEG` | `cy_pp, cy_pn` read POS only; `cy_np, cy_nn` read NEG only ([698–716](../env2_player_reflex_bio2.py#L698)) | no | Keep pair |
| `Y_POS` / `Y_NEG` | `cy_pp, cy_np` read POS; `cy_pn, cy_nn` read NEG | no | Keep pair |
| `X_POS` / `X_NEG` | `NC` only; `W[NC, X_POS] = W[NC, X_NEG] = −K_SHARP` ([569–571](../env2_player_reflex_bio2.py#L569)) | **yes** | **Collapse** → use `bump` to replace NC directly (see below) |
| `X5_POS` / `X5_NEG` | `FC` (sum), **AND** `Wout[0, X5P]=+FRONT_GAIN_MAG`, `Wout[0, X5N]=−FRONT_GAIN_MAG` ([532–533](../env2_player_reflex_bio2.py#L532)) | no (steering uses difference) | **Keep pair** |
| `XE_P` / `XE_N` | `NCR_E` only; both with `−K_SHARP` ([634–635](../env2_player_reflex_bio2.py#L634)) | **yes** | **Collapse** → use `bump` for the `|x10+DRIFT| < THR` test |
| `XW_P` / `XW_N` | `NCR_W` only; both with `−K_SHARP` | **yes** | **Collapse** → use `bump` for `|x10−DRIFT| < THR` |

So three collapses make the cut: `sin_pos/neg`, `x_pos/neg` (and `NC`),
and the `xe_*` / `xw_*` pairs (and `ncr_e` / `ncr_w` predicate
rewrites). **`x5_pos/neg` stays as a pair** — it contributes
opposite-sign steering through `Wout`, so collapsing would silently
disable the front-block corrective steering.

### Step 6a — `sin_sq` (replaces SIN_POS + SIN_NEG)

- `alloc("sin_sq")` instead of `alloc("sin_pos"); alloc("sin_neg")`.
- In `make_activation()`: add `sq_arr` list; apply
  `out[sq_arr, 0] = x[sq_arr, 0] ** 2`.
- Wire `W[SIN_SQ, SIN_N] = 1.0` → output is `sin²(θ)`.
- Retune HV and HH thresholds:
  ```python
  # HV: sin² > 0.70² = 0.49
  W[HV, SIN_SQ] =  K_SHARP / 0.49
  Win[HV, bias_idx] = -K_SHARP
  # HH: sin² < 0.35² = 0.1225
  W[HH, SIN_SQ] = -K_SHARP / 0.1225
  Win[HH, bias_idx] =  K_SHARP
  ```
  The old form used `tanh(0.70) ≈ 0.604` and `tanh(0.35) ≈ 0.336`
  via low-gain relu_tanh — a small approximation error. New form is
  exact; expect a tiny score shift.

### Step 6b — `near_center` via `bump` (replaces X_POS + X_NEG + NC)

Three neurons (X_POS, X_NEG, NC) collapse to one. Current NC fires
when `|X10| < NEAR_CENTER_THR = 0.05`. Using the new bump:

```python
# z = X10 / (2 * NEAR_CENTER_THR) = X10 / 0.1
# bump(z) = max(0, 1 − 4·z²) fires for |z| < 0.5, i.e. |X10| < 0.05  ✓
W[NC, 10] = 1.0 / (2.0 * NEAR_CENTER_THR)    # via bump activation
```

Put `NC` in `bump_arr` (same activation set as Xi_blue).

**Shape difference to be aware of:** old NC was binary (saturated
`relu_tanh`, ~1 when inside, ~0 when outside). New NC is a parabolic
bump — peaks at 1 at `X10=0` and falls smoothly to 0 at `±0.05`. The
`NCV = NC AND HV` AND-gate downstream uses a threshold of 1.5
([line 606](../env2_player_reflex_bio2.py#L606)), which relied on NC
≈ 1 inside the band. With the new bump, NC inside the band is ≥ 0.5
only near the center half of the window.

Two mitigations:
- **Narrow bump scaling**: use `z = X10 / NEAR_CENTER_THR` (support of
  bump becomes `|X10| < NEAR_CENTER_THR/2 = 0.025`). Crisper center,
  but narrower firing band.
- **Post-bump threshold**: retune NCV's bias so `NC + HV > 1.2` rather
  than `> 1.5` fires.

Recommended: go with the first form (z = X10/0.1), plus NCV bias
retune `−K_SHARP * 1.2` instead of `−K_SHARP * 1.5`. Validate
empirically.

### Step 6c — `near_corr_e` / `near_corr_w` via `bump`

Current:
- XE_P/XE_N → NCR_E = cos_big_pos AND `|X10 + DRIFT_OFFSET| < THR`
- XW_P/XW_N → NCR_W = cos_big_neg AND `|X10 − DRIFT_OFFSET| < THR`

Each side collapses to a single bump-centered "near-offset-line"
neuron:
```python
# East line: z_e = (X10 + DRIFT_OFFSET) / 0.1, bump fires for
# |X10 + DRIFT_OFFSET| < 0.05
W[NEAR_E, 10]           = 1.0 / 0.1
Win[NEAR_E, bias_idx]   = DRIFT_OFFSET / 0.1

# West line: z_w = (X10 − DRIFT_OFFSET) / 0.1
W[NEAR_W, 10]           = 1.0 / 0.1
Win[NEAR_W, bias_idx]   = -DRIFT_OFFSET / 0.1
```

Then `NCR_E = cos_big_pos AND NEAR_E` collapses a 3-input relu_tanh
AND into a 2-input one. Same for NCR_W.

### Summary of neuron count change for Step 6

| Net change | Before | After |
|---|---|---|
| 6a | 2 (`sin_pos`, `sin_neg`) | 1 (`sin_sq`) |
| 6b | 3 (`x_pos`, `x_neg`, `near_center`) | 1 (`near_center` bump) |
| 6c | 6 (`xe_p`, `xe_n`, `xw_p`, `xw_n`, unchanged `ncr_e`, `ncr_w`) — but we can collapse xe_*/xw_* (4 neurons) into 2 (`near_e`, `near_w`) used directly in `ncr_e`/`ncr_w` | − 4 |

Net saving: 2 + 2 + 4 = **8 neurons** (before any other steps).

### Risks & verification

- 6a is the safest (just a functional-form change; threshold
  arithmetic is exact).
- 6b changes NCV gate shape — run `_debug_bio2_detail.py` on a trace
  that spends time near the centerline; confirm NCV fires with roughly
  the same frequency.
- 6c: bump width is 0.1 total, matching `2 * NEAR_CENTER_THR`. If
  `DRIFT_OFFSET` is later changed, both `Win` entries must track.
- If any sub-step regresses the score: gate it behind a boolean and
  keep only the ones that hold.

---

## Step 7 — Remove `tt_plus` / `tt_minus`

**Goal:** delete two diagnostic-only neurons.

From the docstring on
[line 418](../env2_player_reflex_bio2.py#L418), the design explicitly
skips TTP/TTM and feeds `cy_*` directly into X20. TTP/TTM are still wired
but unused.

### Edits

1. In `_bio2_indices()`:
   - Delete `alloc("tt_plus")`, `alloc("tt_minus")`.
2. In `reflex_bio2_player()`:
   - Delete aliases `TTP`, `TTM`
     ([lines 287–288](../env2_player_reflex_bio2.py#L287)).
   - Delete the wiring block
     [lines 718–725](../env2_player_reflex_bio2.py#L718).
3. Update layout docstring.

### Verification

- Step-trace must be bit-identical.

---

## Step 8 — Split X20 into `shortcut_steer` and `init_impulse`

**Goal:** make the two semantically-distinct terms of X20 into two
separately-named neurons. Names per decision 5:
`shortcut_steer` and `init_impulse`. Neither carries "20" in the name.

Current X20 wiring
([lines 420–429](../env2_player_reflex_bio2.py#L420)):
```
X20 = |SHORTCUT_TURN|·(CY_PN + CY_NP − CY_PP − CY_NN)    # Part A: turn steering
    + (−SEEDP + SEEDN)                                    # Part B: init correction
Wout[0, 20] = 1.0
```

Output of X20 drives steering directly (one-to-one into
`Wout[0, 20] = 1.0`). After the split, both new neurons feed `Wout`
with weight 1 so the sum is unchanged.

### Edits

1. In `_bio2_indices()`:
   - Add `alloc("shortcut_steer")`.
   - Add `alloc("init_impulse")`.
   - Index `20` becomes an unused slot (no alloc occupies it). Fine
     for correctness; if we later compact the layout, it is a free
     slot to reuse.
2. In `make_activation()`
   [line 201](../env2_player_reflex_bio2.py#L201):
   - Replace `20` in `id_list` with `idx["shortcut_steer"],
     idx["init_impulse"]`.
3. In `reflex_bio2_player()`:
   - Add aliases:
     `SHORTCUT_STEER = idx["shortcut_steer"]`,
     `INIT_IMPULSE   = idx["init_impulse"]`.
   - Replace the block at
     [lines 415–429](../env2_player_reflex_bio2.py#L415):
     ```python
     # ── SHORTCUT_STEER: cy_* quadrant → steering magnitude -------
     W[SHORTCUT_STEER, CY_PN] =  abs(SHORTCUT_TURN)    # cos+, y−
     W[SHORTCUT_STEER, CY_NP] =  abs(SHORTCUT_TURN)    # cos−, y+
     W[SHORTCUT_STEER, CY_PP] = -abs(SHORTCUT_TURN)    # cos+, y+
     W[SHORTCUT_STEER, CY_NN] = -abs(SHORTCUT_TURN)    # cos−, y−

     # ── INIT_IMPULSE: one-shot correction from step-0 seed -------
     # Bio2 analogue of bio1's X23 integrator; SEED_* fires only at
     # step 1 before X24 latches, giving exactly one step of
     # −current_corr_0 into the output.
     W[INIT_IMPULSE, SEEDP] = -1.0
     W[INIT_IMPULSE, SEEDN] =  1.0
     ```
   - Replace `Wout[0, 20] = 1.0` with:
     ```python
     Wout[0, SHORTCUT_STEER] = 1.0
     Wout[0, INIT_IMPULSE]   = 1.0
     ```
4. Update the `X20` block in
   [env2_player_reflex_bio2.md](../env2_player_reflex_bio2.md) to
   describe the two new neurons under their new names.

### Verification

- Output `O` is the sum of both new channels' contributions — must be
  bit-identical to baseline. Step-trace ≥ 1e-9 agreement.
- If a trace diverges, check that neither neuron is accidentally
  dropped from `id_list` in `make_activation()` (both must keep
  identity activation).

---

## Step 9 — Docs refresh

Once all code steps are in, update
[env2_player_reflex_bio2.md](../env2_player_reflex_bio2.md) to reflect:

- New neuron count (Xi_hi/Xi_lo collapse saves 64; dead removals save
  ≤ 6; split of X20 adds 1 → net roughly −69).
- New activation types (`bump`, `sq`) and where they are used.
- Updated layout table (shift of indices after Xi_blue collapse).
- Current score after all steps vs. baseline 14.71.

No code changes in this step.

---

## Summary table

| Step | What | Expected behavior change | Risk |
|---|---|---|---|
| 0 | Baseline snapshot | none | none |
| 1 | Remove X5 / X8 / X9 / X12 | bit-identical | low |
| 2A | Unify activation to `sin` only (bias=π for COS_N) | bit-identical | low |
| 2B | Semantic trig fix (bias=0 / π/2) + downstream audit | physical change | **high** |
| 3 | Remove X23 loop + IS_INIT | bit-identical | low |
| 4 | Xi_blue bump (with optional Xi_red fallback) | small drift | **medium** |
| 5 | X19 gain 1e6 → 500 | bit-identical | low |
| 6a | `sin_sq` (replace sin_pos/neg) | tiny drift (thresholds exact) | low |
| 6b | `near_center` via bump (replace x_pos/neg + NC) | drift; retune NCV bias | medium |
| 6c | `near_e` / `near_w` via bump (replace xe/xw pairs) | drift | medium |
| 7 | Remove tt_plus / tt_minus | bit-identical | low |
| 8 | Split X20 → `shortcut_steer` + `init_impulse` | bit-identical | low |
| 9 | Docs refresh | n/a | none |

Order of application: 0 → 1 → 3 → 5 → 7 → 8 → 2 → 6 → 4 → 9.
Rationale: do the bit-identical cleanups first so each non-trivial
step (2, 4, 6) runs against an already-simplified baseline. Step 4
(new detector shape) is done last before docs because its score drift
is the hardest to predict.

Each step = one commit for clean bisect. Record the score after every
step; halt and re-examine if a step drops score by more than ~0.05.

---

## All decisions locked

- **Step 2: 2A** — unify activation to `sin` only, set COS_N bias to
  `+π` so current output is preserved bit-identically. No downstream
  audit. The naming hazard (`COS_N` outputs `−sin(θ)`) remains; a
  future plan may address it as 2B.
- **Item 4:** `bump(x) = max(0, 1 − 4x²)`, single Xi_blue per ray,
  optional Xi_red fallback if score regresses.
- **Item 5:** `W[19, NCV] = −500`.
- **Item 6:** collapse by pair audit only; X5_pos/neg stays as a pair
  because of its opposite-sign Wout entries.
- **Item 8:** `shortcut_steer` and `init_impulse`.

I will execute step-by-step, pausing for score confirmation after
Step 4 and any 6b/6c sub-step that diverges — but only after you give
the go-ahead to start implementing.
