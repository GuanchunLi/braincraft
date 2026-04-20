# Streamline `env2_player_reflex_bio2`

## 0. Goal and success criteria

Goal: make [env2_player_reflex_bio2.py](../env2_player_reflex_bio2.py) and
its companion [env2_player_reflex_bio2.md](../env2_player_reflex_bio2.md)
easier to read without changing behaviour.

Success criteria (all must hold after every phase below):

1. `python braincraft/env2_player_reflex_bio2.py` still prints
   `Final score (distance): 14.71 +/- 0.00`.
2. `python braincraft/_debug_bio2.py --steps 120 --stride 20` runs to
   completion; every printed bio2 line matches the pre-refactor trace
   (same `O`, `pos`, `E`).
3. `python braincraft/_debug_bio2_detail.py --steps 120` runs to
   completion; all per-neuron fields are numerically identical to the
   pre-refactor trace (after the renamed `bio2_idx` lookups).
4. The description file `env2_player_reflex_bio2.md` lists every helper
   neuron in the same order that `_bio2_indices` allocates it, and every
   equation uses the `(t+1) / (t)` time convention.

Rule of thumb: no phase is a "silent refactor". After each phase, re-run
all three commands above. Ship the plan one phase at a time so a
behaviour regression points at a single, small diff.

---

## 1. Preparation

Before any code change:

1.1. Capture a baseline trace and keep it on disk:

```powershell
python braincraft\_debug_bio2.py        --steps 120 --stride 20 > plans\baseline_bio2_coarse.txt
python braincraft\_debug_bio2_detail.py --steps 120             > plans\baseline_bio2_detail.txt
python braincraft\env2_player_reflex_bio2.py                    > plans\baseline_bio2_score.txt
```

These files are the ground truth we diff against after each phase.
(They are local scratch files; do not commit.)

1.2. Identify every hard-coded integer neuron index in the repo that
refers to bio2 slots, so none are missed during renumbering.

```powershell
Select-String -Path braincraft\*.py -Pattern "X_new\[|X\[\d|W\[\d|Win\[\d|Wout\[0, \d"
```

Known offenders today:

- [_debug_bio2_detail.py:52-55](../_debug_bio2_detail.py#L52-L55) reads
  `X_new[10,0]`, `X_new[11,0]`, `X_new[19,0]`, `X_new[22,0]` directly.
- [env2_player_reflex_bio2.py](../env2_player_reflex_bio2.py) itself uses
  `W[6,...]`, `W[7,...]`, `W[10,...]`, `W[11,...]`, `W[13,...]`,
  `W[14,...]`, `Win[14,...]`, `W[22,...]`, `Win[22,...]`, `W[24,...]`,
  `Win[24,...]`, plus the reward block `W[15..18,...]` / `Win[15..18,...]`.

All of these must be driven off the `idx` dict after Phase 3, otherwise
the renumber silently breaks the wiring.

1.3. One-line check that no other player imports bio2 constants:

```powershell
Select-String -Path braincraft\*.py -Pattern "from env2_player_reflex_bio2|import env2_player_reflex_bio2"
```

Only `_debug_bio2.py`, `_debug_bio2_detail.py` should match. (Verified
2026-04-20; keep as a sanity check before committing.)

---

## 2. Phase A — Activation simplification (user's point 1)

Audit every neuron whose activation is currently `relu_tanh` and ask
whether thresholding is doing real work. Keep `relu_tanh` only when the
neuron is a genuine threshold / AND / OR / self-latch; otherwise
downgrade to `relu` (if the signal is non-negative) or `identity`.

Decision table (proposed; verify each with a one-neuron swap + baseline
diff):

| Neuron(s) | Current | Proposed | Reason |
| --- | --- | --- | --- |
| `X0..X4` reflex | `relu_tanh` | **keep** | Genuine threshold on hit/proximity. |
| `X15` (energy ramp) | `relu_tanh` | **`relu`** | `K·energy` with `K=0.005`; energy is ≥0, value is O(≤1), `relu_tanh(z) ≈ z` throughout, `relu` matches exactly and is cheaper/clearer. |
| `X16` (armed latch) | `relu_tanh` | **keep** | Self-recurrent gain-10 latch, needs saturation. |
| `X17` (pulse) | `relu_tanh` | **keep** | Sharp detector with huge negative bias. |
| `X18` (reward latch) | `relu_tanh` | **keep** | Self-recurrent latch. |
| `X24` seeded flag | `relu_tanh` | **keep** | Clean "saturates to 1 after step 0" semantics. (Swapping to `identity` with bias 1 also works but reads as "why 1?" — keep.) |
| `L_EV, R_EV` | `relu_tanh` | **`identity`** (make the "sum" real) | User decision: replace the saturating `relu_tanh` with `identity` so `L_EV` / `R_EV` literally equal `sum(xi_blue[...])`. Requires a dedicated validation pass (see Phase A.2 below) because DLEFT/DRIGHT downstream may behave differently when both sides see color simultaneously. |
| `DLEFT, DRIGHT` | `relu_tanh` | **keep** | Binary pulses with `-0.2·K_SHARP` margin. |
| `EVIDENCE` | `identity` | **keep** | Signed integrator. |
| `TP, TN` | `relu_tanh` | **keep** | Threshold. |
| `FS_P, FS_N` | `relu_tanh` | **keep** | Self-latch. |
| `X5P, X5N` | `relu_tanh` | **keep** | Threshold gate. |
| `COS_POS/NEG, Y_POS/NEG` | `relu_tanh` | **keep** | Sign extractors. |
| `NEAR_C/E/W` | `bump` | **keep** | Compact support is load-bearing. |
| `HH, FC` | `relu_tanh` | **keep** | Threshold predicate. |
| `COS_BIG_*, COS_SMALL` | `relu_tanh` | **keep** | Threshold. |
| `NCR_{E,W,C}, NEAR_CORR` | `relu_tanh` | **keep** | AND / OR. |
| `TSC` | `relu_tanh` | **keep** | 4-way AND with refractory. |
| `ON22, IST, ISA` | `relu_tanh` | **keep** | Phase predicates. |
| `CY_{PP,PN,NP,NN}` | `relu_tanh` | **keep** | 3-way AND. |
| `SEEDP, SEEDN` | `relu_tanh` | **keep** | Gated one-shot pulses. |
| `xi_blue[*]` | `bump` | **keep** | Color detector. |
| `X6` | `clip_a` | **keep** | Steering output, exact clip required. |
| `X7, X10, X11, X13, EVIDENCE, SHORTCUT_STEER, INIT_IMPULSE` | `identity` | **keep** | Accumulators / signed passthrough. |
| `X22` countdown | `relu` | **keep** | Counter that exceeds 1. |

Concrete changes from this phase, split into two commits so they can
be bisected independently:

**Phase A.1 — `X15` → `relu`.**
Move `X15` from the default `relu_tanh` bucket to the `relu_list` in
`make_activation`. Expected: byte-identical traces and score, because
`relu_tanh(z) = relu(z)` for `z ∈ [0, 1]` and `K·energy` stays well
inside that range. Re-run all three baselines; if they diverge, either
`energy` escapes `[0, 200]` somewhere (unlikely but possible), or the
override list has a typo.

**Phase A.2 — `L_EV` / `R_EV` → `identity`.**
Add both to `id_list` in `make_activation`. Since the downstream
consumer is `DLEFT / DRIGHT = relu_tanh( k_sharp · (L_EV − R_EV) − 0.2·k_sharp − (latch inhibition) )`,
the sign and AND semantics are preserved when only one side sees
color. When *both* sides see color simultaneously, the raw sums no
longer saturate at 1, so `DLEFT` / `DRIGHT` get a steeper drive — the
test matters here.

Validation protocol for A.2:

1. Diff the new `_debug_bio2.py` and `_debug_bio2_detail.py` traces
   against the post-A.1 baseline.
2. If the final score remains `14.71 +/- 0.00` and the trajectory
   (position log) is identical, keep the change and update the `.md`
   to say "`L_EV` and `R_EV` are identity — literal sums — so the
   equation in §4.5 is exact".
3. If the score or trajectory changes but the new score is ≥ 14.71,
   keep the change and record the new number in the doc footer.
4. If the score drops, **do not auto-revert**. Flag the divergence to
   the user with the first step at which traces diverge, the value of
   `L_EV / R_EV / DLEFT / DRIGHT` at that step, and ask whether to
   revert or to compensate downstream (e.g., scale the `k_sharp` in
   `DLEFT / DRIGHT`).

---

## 3. Phase B — Drop unused neurons (user's point 2, part 1)

Slots to delete outright (no reads, no non-trivial writes):

| Slot | Today | Action |
| --- | --- | --- |
| `X5` | "unused legacy" | Delete. |
| `X8, X9` | "unused legacy" | Delete. |
| `X12` | "unused legacy" | Delete. |
| `X14` | Writes `Win[14, L_idx]` and `Win[14, R_idx]` but nothing reads `X14`. | Delete (remove both `Win[14, ...]` lines). |
| `X19` | Former step counter. | Delete. Also remove the `X_new[19, 0]` read in [_debug_bio2_detail.py:54](../_debug_bio2_detail.py#L54). |

All of the above simply disappear in Phase C renumbering; there is no
per-slot zeroing to do in this phase. The point of calling this out
separately is that the **description file** currently lists them as
"unused" — Phase E removes those rows.

Grep sweep to confirm nothing else references the deleted slots:

```powershell
Select-String -Path braincraft\*.py,braincraft\*.md -Pattern "\bX5\b|\bX8\b|\bX9\b|\bX12\b|\bX14\b|\bX19\b"
```

(Expected remaining hits after deletion: `X5P / X5N` names, which are
different neurons and must be renamed in Phase D to avoid confusion —
see §5.)

---

## 4. Phase C — Renumber into the requested order (user's point 2, part 2)

Target ordering (user's list, made explicit):

1. Reflex features
2. Steering `dtheta`
3. Direction accumulator
4. `(x, y)` position
5. Initial heading correction
6. Reward circuit
7. Shortcut circuit
8. Color evidence circuit (includes the color-gated front block)

Dense layout (`n=1000`, only indices 0..121 populated):

```
Group 1 — Reflex features                              (relu_tanh)
   0  hit_feat             (was X0)
   1  prox_left            (was X1)
   2  prox_right           (was X2)
   3  safe_left            (was X3)
   4  safe_right           (was X4)

Group 2 — Steering                                     (clip_a)
   5  dtheta               (was X6)

Group 3 — Direction accumulator                        (identity)
   6  dir_accum            (was X7)

Group 4 — Position                                     (identity)
   7  pos_x                (was X10)
   8  pos_y                (was X11)

Group 5 — Initial heading correction block
   9  head_corr            (was X13, identity)
  10  seeded_flag          (was X24, relu_tanh)
  11  seed_pos             (relu_tanh)
  12  seed_neg             (relu_tanh)

Group 6 — Reward circuit
  13  energy_ramp          (was X15, relu  — see Phase A)
  14  armed_latch          (was X16, relu_tanh)
  15  reward_pulse         (was X17, relu_tanh)
  16  reward_latch         (was X18, relu_tanh)

Group 7 — Shortcut circuit
  17  sc_countdown         (was X22, relu)
  18  shortcut_steer       (identity)
  19  init_impulse         (identity)
  20  cos_n                (sin)
  21  sin_n                (sin)
  22  sin_sq               (square)
  23  cos_pos              (relu_tanh)
  24  cos_neg              (relu_tanh)
  25  y_pos                (relu_tanh)
  26  y_neg                (relu_tanh)
  27  cos_big_pos          (relu_tanh)
  28  cos_big_neg          (relu_tanh)
  29  cos_small            (relu_tanh)
  30  near_c               (was near_center, bump)  — see §5 rename
  31  near_e               (bump)
  32  near_w               (bump)
  33  ncr_e                (relu_tanh)
  34  ncr_w                (relu_tanh)
  35  ncr_c                (relu_tanh)
  36  near_cr              (was near_corr, relu_tanh)  — see §5.3 rename
  37  heading_horiz        (relu_tanh)
  38  front_clear          (relu_tanh)
  39  trig_sc              (relu_tanh)
  40  on_countdown         (was on_22, relu_tanh)
  41  is_turn              (relu_tanh)
  42  is_app               (relu_tanh)
  43  cy_pp                (relu_tanh)
  44  cy_pn                (relu_tanh)
  45  cy_np                (relu_tanh)
  46  cy_nn                (relu_tanh)

Group 8 — Color evidence & color-gated front block
  47  front_block_pos      (was x5_pos, relu_tanh)  — see §5 rename
  48  front_block_neg      (was x5_neg, relu_tanh)
  49  l_ev                 (relu_tanh)
  50  r_ev                 (relu_tanh)
  51  dleft                (relu_tanh)
  52  dright               (relu_tanh)
  53  evidence             (identity)
  54  trig_pos             (relu_tanh)
  55  trig_neg             (relu_tanh)
  56  fs_pos               (relu_tanh)
  57  fs_neg               (relu_tanh)
  58..121  xi_blue[0..63]  (bump)
```

Rationale for sub-ordering inside Group 7: upstream-first. `cos_n`,
`sin_n` before derived magnitudes; the `near_*` / `ncr_*` ladder before
its `near_corr` consumer; predicates (`heading_horiz`, `front_clear`)
before `trig_sc`; `trig_sc` before the phase flags it gates
(`on_countdown`, `is_turn`, `is_app`); phase flags before the quadrant
ANDs that multiply them. Reading `_bio2_indices` top-to-bottom now
mirrors data flow.

Rationale for Group 8 order: `front_block_{pos,neg}` are consumed by
`front_clear` (upstream of `trig_sc`) and by `Wout` directly, so they
have to live in the layout. Putting them at the top of the color group
keeps the xi_blue tail contiguous; their `FS_*` inputs follow. The
xi_blue bank sits at the bottom so the helper block is compact.

Implementation notes:

- Rewrite `_bio2_indices` from scratch in the new order. Keep it as a
  single `alloc` cascade — it becomes the layout spec.
- Replace *every* hard-coded integer index in the builder with a lookup
  from `idx`. In particular:
  - `W[6, ...]` → `W[idx["dtheta"], ...]`
  - `W[7, 7]`, `W[7, 6]` → `W[idx["dir_accum"], idx["dir_accum"]]` etc.
  - `W[10, ...]`, `W[11, ...]` → `idx["pos_x"]`, `idx["pos_y"]`
  - `W[13, 13]`, `W[13, SEEDP]`, `W[13, SEEDN]` → `idx["head_corr"]`
  - `W[22, 22]`, `Win[22, bias_idx]`, `W[22, TSC]` → `idx["sc_countdown"]`
  - `W[24, bias_idx]` is wrong in the current code — it should be
    `Win[24, bias_idx]`. Verify in the current file; this is the
    `Win[24, bias_idx] = 10.0` line at
    [env2_player_reflex_bio2.py:379](../env2_player_reflex_bio2.py#L379)
    (already correct). Just re-verify after renumbering.
  - Reward block `W/Win[15..18, ...]` → `idx["energy_ramp"]`, `idx["armed_latch"]`, `idx["reward_pulse"]`, `idx["reward_latch"]`.
  - `for i in range(5): W[i, ISA]` loop that silences reflex in
    approach phase → iterate over an explicit list of the five reflex
    `idx` entries.
  - `for j in range(n): if Wout[0, j] != 0: W[6, j] = Wout[0, j]` tail
    block → `W[idx["dtheta"], j] = Wout[0, j]`.
- Update `make_activation` so every override list is built from `idx`
  lookups. No bare integers except via `idx`. In particular replace:
  - `id_list = [7, 10, 11, 13, 14, idx["shortcut_steer"], idx["init_impulse"], idx["evidence"]]`
    → drop `14` (deleted in Phase B), use `idx` for the rest.
  - `relu_list = [22]` → `relu_list = [idx["sc_countdown"]]`; add
    `idx["energy_ramp"]` per Phase A.
  - The `out[6, 0] = np.clip(...)` line → `out[idx["dtheta"], 0] = ...`.
- Update `_debug_bio2_detail.py` to read the four hard-coded indices
  through the idx map: call `bio2._bio2_indices(n_rays=p, use_xi_red=False)`
  once and read `X_new[idx["pos_x"], 0]`, etc. Drop the `X_new[19, 0]`
  read entirely (X19 is gone).
- The `use_xi_red` code path is dead under the accepted
  `USE_XI_RED_FALLBACK=False` and is being **deleted** (user decision).
  Remove:
  - the `use_xi_red=True` branch inside `_bio2_indices` (the
    `xi_red_start` / `xi_red_stop` entries);
  - the `bump_list.extend(range(idx["xi_red_start"], idx["xi_red_stop"]))`
    override inside `make_activation`;
  - the `Win[idx["xi_red_start"] + r, ...]` loop block in the builder;
  - the two `if "xi_red_start" in idx` branches in the `L_EV` / `R_EV`
    summing loop;
  - the `USE_XI_RED_FALLBACK` module constant and the `use_xi_red=`
    argument of `_bio2_indices` (it becomes unconditional
    `n_rays`-only).

Validation after Phase C: all three baselines must still match byte-for-byte
(per-step `O` values are deterministic given the same seed). This is the
highest-risk phase of the refactor.

---

## 5. Phase D — Naming / casing / notation (user's point 3)

### 5.1 Constants → lowercase (both `.py` and `.md`)

In-file rename:

| Old | New |
| --- | --- |
| `K_SHARP` | `k_sharp` |
| `STEP_A` | `step_a` |
| `SHORTCUT_TURN` | `shortcut_turn` |
| `SIN_HORIZ_THR` | `sin_horiz_thr` |
| `SIN_VERT_THR` | **delete** (dead; no consumers) |
| `COUNTER_THR` | **delete** (dead; no consumers) |
| `NEAR_CENTER_THR` | `near_c_thr` (matches the `NEAR_C` rename; see §5.3) |
| `DRIFT_OFFSET` | `drift_offset` |
| `TURN_STEPS` | `turn_steps` |
| `APPROACH_STEPS` | `approach_steps` |
| `SC_TOTAL` | `sc_total` |
| `COLOR_EVIDENCE_THRESHOLD` | `color_evidence_thr` (also shorter) |
| `FRONT_GAIN_MAG` | `front_gain_mag` |
| `GATE_C` | `gate_c` |
| `USE_XI_RED_FALLBACK` | **delete** (see §4 removal list) |

Note: this diverges from PEP 8 (which asks UPPER_SNAKE for module-level
constants). Accepted by the user — consistency between the code and the
math in `env2_player_reflex_bio2.md` matters more than PEP 8 here. Add a
one-line module docstring noting the convention ("constants are
`snake_case`; hidden-state neuron names are `SHOUTING_SNAKE`"), so future
contributors don't "fix" it.

Before running the `replace_all`, grep for the token across the repo to
confirm the bio2 file is the only consumer; the grep from §1.3 already
did this.

### 5.2 Equation convention: `_prev` → `(t)` / `(t+1)`

Rule for the `.md` file: for each neuron equation, write the LHS as
`NAME(t+1)` and every RHS occurrence of a *state* neuron as `NAME(t)`.
Input wires `prox[i]`, `color[i]`, `hit`, `energy` are the current-step
input and keep the plain name (no time tag needed — they are `I[t]` by
construction).

Worked example (from §4.9 of the current doc):

Before:

```text
COS_POS  = relu_tanh(  K_SHARP · COS_N_prev )
```

After:

```text
COS_POS(t+1) = relu_tanh( k_sharp · COS_N(t) )
```

Apply the same transform to every equation in §4.1–§4.19 of the doc.
The `X[t+1] = f(Win·I[t] + W·X[t])` masthead in §1 already uses the
new convention, so the document becomes internally consistent.

Scope: doc only. The Python comments in `env2_player_reflex_bio2.py`
that mention `_prev` should also be updated (search-and-replace) for
consistency — one-line cost, and it keeps comments aligned with the
doc's convention.

### 5.3 Neuron-name consistency

- `NEAR_CENTER` → `NEAR_C` (matches `NEAR_E` / `NEAR_W`). Update the
  `idx` key (`"near_center"` → `"near_c"`) and all aliases
  (`NC = idx["near_c"]`). Update the `.md` file. Note: this also
  drives the `NEAR_CENTER_THR` → `near_c_thr` rename in §5.1.
- `x5_pos` / `x5_neg` → `front_block_pos` / `front_block_neg`. The
  `X5` prefix is misleading now that the slot number is different
  (47/48 in the new layout) and there is no plain `X5` neuron either
  (Phase B deleted the legacy slot). Update the `idx` keys, aliases
  (`X5P` → `FBP`, `X5N` → `FBN`), and the `.md` file.
- `on_22` → `on_countdown`. The `22` in the name tracks the old slot
  number, which no longer exists after Phase C. Update `idx`, alias
  (`ON22` → `ONC`), `.md`.
- `NEAR_CORR` → `NEAR_CR`. Matches the `NEAR_E` / `NEAR_W` / `NEAR_C`
  two-letter suffix pattern. Update `idx` key (`"near_corr"` →
  `"near_cr"`), alias (`NEAR_CORR` → `NEAR_CR`), all `W[NEAR_CORR, ...]`
  wiring (three lines), and the `.md` §4.15 subsection heading +
  equations.

Other names (`L_EV`, `R_EV`, `DLEFT`, `DRIGHT`, `TSC`, `HH`, `FC`,
`IST`, `ISA`, `CY_*`, `COS_POS/NEG`, `Y_POS/NEG`, `FS_P/N`, `NCR_*`,
`SEEDP/N`, `SIN_SQ`, `COS_BIG_*`, `COS_SMALL`) are already consistent
and stay as-is.

### 5.4 `x20_val` comment and other stale references

Grep for stale comments that still say `X20`, `X22`, `X24`, `X5` and
update them to refer to the new names or drop the slot number
entirely:

```powershell
Select-String -Path braincraft\env2_player_reflex_bio2.py -Pattern "X20|X22|X24|X5"
```

---

## 6. Phase E — Rewrite the description file

With the code finalized through Phase D, the description
`env2_player_reflex_bio2.md` is rewritten to match:

- **§2 Activation library**: update the `relu` row to list `X19` (now
  `sc_countdown`) *and* `energy_ramp` (the new `relu` entry from
  Phase A). Remove `X19` mention.
- **§3 Input vector and fixed slots**: replace the two slot tables
  (the high-level one and the helper-range one) with a single table
  that lists *every* populated slot in the new order, with its name,
  index, activation, and a one-line role. No "unused" rows.
- **§4 Main circuits**: reorder sub-sections to match the group order
  in §4 of this plan (Reflex → Steering → Direction → Position →
  Init-heading → Reward → Shortcut → Color-evidence-and-front-block).
  Within each sub-section, order neurons in the same upstream-first
  order used by `_bio2_indices`.
- **Notation**: §5.2 conversion of `_prev` → `(t)` convention; §5.1
  constant casing.
- **Dead-subsection cleanup**: §4.12 currently contains a paragraph
  explaining removed `SCI`/`ES` neurons — condense to one sentence
  *in the trig_sc subsection*, or move to a final "Historical notes"
  footer. The current midsection placement interrupts the data-flow
  narrative.
- **Score footer**: keep `Accepted score on 2026-04-17: 14.71 +/- 0.00`
  and add the re-verification date once the refactor lands.

---

## 7. Phase ordering and checkpoints

Apply phases in this order and re-run the three validation commands
(§0) at every checkpoint. Expect byte-identical `O` values through
every phase — any divergence is a bug.

| # | Phase | Behaviour-preserving? | Validation |
| --- | --- | --- | --- |
| 1 | Preparation (§1) | N/A | baseline captured |
| 2 | Phase A: `X15` → `relu` | yes (values equivalent for `z ∈ [0, 1]`) | diff vs baseline |
| 3 | Phase B: drop unused slots | yes | diff vs baseline |
| 4 | Phase C: renumber + idx-ify wiring | yes | diff vs baseline — highest risk |
| 5 | Phase D: naming/notation | yes (identifier churn only) | diff vs baseline |
| 6 | Phase E: rewrite doc | doc-only | manual read-through |

Commit per phase. Don't squash — a regression bisect is much easier
with five small commits than one large one.

---

## 8. Decisions (resolved 2026-04-20)

1. **`use_xi_red` code path** → **delete** (see §4 removal list).
2. **Dead constants `SIN_VERT_THR`, `COUNTER_THR`** → **delete**
   (see §5.1).
3. **`L_EV` / `R_EV` activation** → change to `identity` so the
   documented "sum" is literal; test per the Phase A.2 validation
   protocol in §2.
4. **Python constant casing (lowercase)** → **confirmed**; rename in
   both `.py` and `.md` per §5.1.
5. **Neuron rename scope** → four renames confirmed:
   `NEAR_CENTER` → `NEAR_C`, `x5_{pos,neg}` → `front_block_{pos,neg}`,
   `on_22` → `on_countdown`, **`NEAR_CORR` → `NEAR_CR`** (added per
   user request, see §5.3).

---

## 9. Risks

- **Score regression during Phase C.** The layout has ~50 `W` / `Win`
  writes referencing hard-coded indices. Missing one (e.g., forgetting
  to route `W[6, j] = Wout[0, j]` through `idx["dtheta"]`) silently
  decouples `X6` from the readout and may still produce a *plausible*
  but wrong score. Mitigation: compare the full `_debug_bio2_detail`
  trace, not just the final score.
- **Readout (`Wout`) rows are indexed too.** `Wout[0, SHORTCUT_STEER]`,
  `Wout[0, INIT_IMPULSE]`, `Wout[0, X5P]`, `Wout[0, X5N]` all need the
  renumbered `idx` values — easy to miss because they sit outside the
  wiring block.
- **Activation-function `make_activation` ordering matters.** The
  `relu_tanh` default is applied *first*, then the overrides. Any
  neuron that should use an override but is accidentally omitted from
  `id_arr` / `relu_arr` / `sin_arr` / `square_arr` / `bump_arr` will
  silently fall back to `relu_tanh`. Validate by printing
  `sorted(id_arr | sin_arr | square_arr | relu_arr | bump_arr | {idx["dtheta"]})`
  against the expected set once from the new layout.
- **Hidden dependency on the slot-5 hole.** Nothing currently reads
  `X5` but if any random-access debug utility does, deleting the slot
  and renumbering breaks it. Phase 1 grep covers `.py`; if any other
  notebook or script we don't see has a hard-coded `5`, it will need
  fixing too. Worth a quick `git grep` of the full repo before Phase C.
