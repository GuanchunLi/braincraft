# Reflex Bio2 Streamline Plan

## Summary

- Keep the existing phase structure: Preparation, then Phases A through E.
- Verified locally on April 20, 2026: the current fixed command prints `Final score (distance): 14.71 +/- 0.00`.
- Acceptance for code phases is **score-only**: deterministic trace drift is allowed if the fixed evaluation score does not drop below `14.71`. Baseline traces remain diagnostic, not blocking.
- Scope stays limited to the bio2 controller, its detail debugger, and the bio2 markdown description. `bio1`, the env2 evaluator, and the trajectory plotter are out of scope unless a rename breaks them.

## Interface Changes

- `_bio2_indices` becomes the single layout spec for **every live bio2 slot**, including the low fixed slots; the builder and `_debug_bio2_detail.py` should stop depending on bare bio2 neuron integers.
- `_bio2_indices(n_rays, use_xi_red=False)` becomes `_bio2_indices(n_rays)`. Delete the dead `xi_red_*` allocator path and the `USE_XI_RED_FALLBACK` constant.
- Final key/name cleanup:
  - `near_center -> near_c`
  - `x5_pos/x5_neg -> front_block_pos/front_block_neg`
  - `on_22 -> on_countdown`
  - `near_corr -> near_cr`
- Module constants become lowercase in both code and docs (`k_sharp`, `step_a`, `shortcut_turn`, `near_c_thr`, etc.); hidden-state aliases may remain uppercase if they read better in equations/comments.

## Phase Plan

### 1. Preparation

- Capture three local baseline files under `braincraft/plans/`: score, coarse trace, and detail trace. Treat them as scratch; do not commit them.
- Freeze a migration checklist with two greps:
  - direct imports of `env2_player_reflex_bio2`
  - hard-coded bio2 slot references in code/docs/debug tools
- Record the current baseline score floor as `14.71`.

### 2. Phase A: Activation simplification

- **A.1**: move `energy_ramp` (current `X15`) from default `relu_tanh` to `relu`.
- **A.2**: move `l_ev` and `r_ev` from `relu_tanh` to `identity`.
- After each subphase, run the fixed score command. If the score drops below `14.71`, revert **that subphase only** and continue with the rest of the plan.
- Keep trace diffs for diagnosis, especially for `l_ev/r_ev -> dleft/dright -> evidence`, but do not make trace identity a gate.

### 3. Phase B: Dead state and dead path removal

- Remove the genuinely live dead code before renumbering:
  - delete `X14` wiring
  - delete `X19` handling and the `X19` read in the detail debugger
  - delete the entire `use_xi_red` fallback path end-to-end
- Treat `X5`, `X8`, `X9`, and `X12` as documentation/comment cleanup only; there is no meaningful live wiring left to remove for those slots in the current file.
- Do not renumber yet; the point of Phase B is to shrink the Phase C surface area while preserving behavior.

### 4. Phase C: Renumber and idx-ify

- First update `_debug_bio2_detail.py` so all bio2 reads come from `_bio2_indices(...)` by name; after this step it should survive slot renumbering without edits.
- Rewrite `_bio2_indices` in the final dense order:
  1. reflex
  2. steering
  3. direction
  4. position
  5. initial-heading correction
  6. reward
  7. shortcut
  8. color-evidence/front-block
- Within the shortcut and color groups, keep the current draft's upstream-first ordering so dataflow still reads top-to-bottom.
- Replace all bare bio2 neuron integers in the controller, activation builder, and `Wout -> dtheta` copy with `idx[...]`. Input-column integers such as `hit_idx`/`energy_idx`/`bias_idx` can stay numeric aliases.

### 5. Phase D: Naming and notation cleanup

- Apply the final neuron/key/alias renames and switch module constants to lowercase.
- Remove dead constants `sin_vert_thr` and `counter_thr`.
- Update inline comments and debug labels to use the final names and `(t)/(t+1)` notation, but defer the full markdown rewrite to Phase E.

### 6. Phase E: Rewrite the description doc

- Rewrite the markdown so it matches the final allocator order, final activation library, final names, and `(t+1) = f(..., t)` convention.
- Replace the split slot tables with one complete live-slot table in allocator order.
- Keep the historical accepted-score footer and add the streamline re-verification date.

## Test Plan

- After every code phase, run:
  - `python braincraft/env2_player_reflex_bio2.py`
- After Phases A, C, and D, also run:
  - `python braincraft/_debug_bio2.py --steps 120 --stride 20`
  - `python braincraft/_debug_bio2_detail.py --steps 120`
- Acceptance rule:
  - score command succeeds
  - printed score is deterministic and `>= 14.71`
  - debug scripts complete without stale-slot failures
- Use the baseline traces to localize where behavior changed, but only fail a phase automatically on score regression or broken tooling.
- Final manual check: the markdown helper order matches `_bio2_indices`, and all equations use the `(t+1)/(t)` convention consistently.

## Assumptions

- Score-only acceptance is intentional for this round.
- The current direct import surface is only the two bio2 debug scripts; no other player depends on bio2 allocator internals.
- Execute one phase at a time and keep the diffs separable for bisecting.
