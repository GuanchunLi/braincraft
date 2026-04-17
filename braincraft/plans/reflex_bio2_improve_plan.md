# Reflex Bio2 Simplification Plan

## Summary

- Rewrite the existing plan markdown first so it matches the actual execution
  scope, sequencing, and acceptance gates for this round.
- Keep `bio1`, the env2 evaluation harness, and `n=1000` unchanged.
- Use the current controller as the baseline: fixed local run confirmed `14.71`
  on April 17, 2026.
- Apply trig option **2A** only: remove the `cos` activation branch, preserve
  current outputs, and document the rotated coordinate semantics instead of
  attempting the full semantic trig rewrite.
- Because this round is **Aggressive Simplify**, keep structural
  simplifications even with a small measured drop, but cap the accepted final
  fixed-harness score floor at **14.60**.

## Implementation Changes

### 1. Planning and observability first

- Rewrite this file so it reflects the actual scope of the implementation round.
- Harden [`_debug_bio2_detail.py`](../_debug_bio2_detail.py) so it:
  - reads optional neurons by name,
  - tolerates removed allocator keys,
  - reports the combined shortcut contribution instead of assuming raw `X20`
    is the only shortcut channel.
- Leave [`_debug_bio2.py`](../_debug_bio2.py) unchanged.

### 2. Safe controller cleanup and naming

- Remove dead low-slot diagnostics: drop the unused `X5` and `X12` wiring, and
  clean up stale doc/comment references to `X8`, `X9`, and `X12`.
- Remove the unused `X23` depletion loop and delete `x23p`, `x23n`, and
  `is_init`.
- Remove `tt_plus` and `tt_minus`; keep the direct `cy_* -> shortcut` path only.
- Split the current shortcut output into named fixed-slot channels without
  creating extra live state:
  - `shortcut_steer = 20`
  - `init_impulse = 23` (reusing the dead `X23` slot)
- Apply trig **2A**:
  - keep `SIN_N` at bias `pi/2`
  - change `COS_N` to the `sin` activation with bias `pi`
  - remove the `cos` activation branch from the pointwise activation builder
  - update comments/docs so the preserved rotated frame is explicit
- Defer the `X19` reset-gain reduction; the proposed `-500` change is cosmetic
  and not justified strongly enough without a full-range measurement.

### 3. Activation-library simplification pass

- Add a `square` activation and replace `sin_pos/sin_neg` with a single
  `sin_sq`; retune `heading_vert` and `heading_horiz` against `sin^2`.
- Add a `bump` activation and replace `Xi_hi/Xi_lo` with a single `xi_blue`
  bank by default.
- Only add an `xi_red` fallback bank if the blue-only version falls below the
  accepted score floor.
- Replace `x_pos/x_neg/near_center` with one bump-based `near_center`, and
  retune `nc_and_hv` so the counter reset still fires in the intended corridor
  crossing.
- Replace `xe_pos/xe_neg` and `xw_pos/xw_neg` with bump-based `near_e` and
  `near_w`, keeping `ncr_e`, `ncr_w`, `ncr_c`, and `near_corr` topology intact.
- Update `_bio2_indices()` to match the final accepted design, including the
  fixed-slot aliases for `shortcut_steer` and `init_impulse`.

## Interface / Naming Changes

- `_bio2_indices()` should stop exposing deleted keys from removed helper pairs
  and dead diagnostics.
- New named keys should include `shortcut_steer`, `init_impulse`, `sin_sq`,
  `xi_blue_*`, `near_e`, and `near_w`.
- `xi_red_*` is conditional: add it only if the fallback is actually retained.
- The detail debug output should become name-based and resilient to missing
  optional keys.

## Test Plan

1. Capture the baseline with:
   ```powershell
   python braincraft\_debug_bio2.py --steps 400 --stride 20
   python braincraft\env2_player_reflex_bio2.py
   ```
2. After the safe cleanup cluster, require step-trace agreement to 4 decimals
   and the same fixed-harness score.
3. After each experimental simplification (`sin_sq`, `xi_blue`, `near_center`,
   `near_e/w`), rerun the trace and score.
4. Keep an experimental substep only if:
   - the fixed-harness score stays at or above **14.60**
   - the trace does not show obvious shortcut-timing failure or wall-hit
     oscillation
5. Final gate:
   ```powershell
   python braincraft\_debug_bio2_detail.py --steps 120
   python braincraft\env2_player_reflex_bio2.py
   ```
   The detail script must run without missing-key errors, and the docs must
   match the final accepted neuron set.

## Assumptions

- Scope choice: **Aggressive Simplify**.
- Trig choice: **2A Preserve Output**.
- No changes to `bio1`, `challenge_2.py`, or the env2 environment.
- Fixed-slot reuse is preferred over adding fresh named neurons when an old dead
  slot already exists.
- The final design may keep a small deterministic score drop, but not below
  **14.60** on the current fixed evaluation command.
