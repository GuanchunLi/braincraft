"""Ablation runner for reflex_bio2.

Two kinds of ablation are supported:

  1. Reflex slots 0..4: zero the Wout weight (and its copy into W[dtheta, :]).
     Disables that feature's contribution to steering.
  2. Internal edges: zero a single W[dst, src] entry. Used to probe the
     corridor / shortcut circuit (e.g. near_c -> ncr_c, ncr_c -> near_cr).
"""

import numpy as np
import time

from bot import Bot
from environment_2 import Environment
from challenge_2 import evaluate, train
from env2_player_reflex_bio2 import reflex_bio2_player, _bio2_indices


SLOTS = {
    "hit_feat":   0,
    "prox_left":  1,
    "prox_right": 2,
    "safe_left":  3,
    "safe_right": 4,
}


def ablate(model, names):
    Win, W, Wout, warmup, leak, f, g = model
    Wout = Wout.copy()
    W = W.copy()
    for name in names:
        idx = SLOTS[name]
        Wout[0, idx] = 0.0
        # Also zero the Wout-row copy into W[dtheta, :].
        # dtheta index = 5 per _bio2_indices.
        W[5, idx] = 0.0
    return (Win, W, Wout, warmup, leak, f, g)


def ablate_edges(model, edges, n_rays=64):
    """Zero individual W[dst, src] entries, identified by name pairs."""
    Win, W, Wout, warmup, leak, f, g = model
    W = W.copy()
    idx = _bio2_indices(n_rays)
    for dst, src in edges:
        W[idx[dst], idx[src]] = 0.0
    return (Win, W, Wout, warmup, leak, f, g)


def run(label, ablate_names=None, edges=None, rewire=None, seed=12345):
    np.random.seed(seed)
    base = train(reflex_bio2_player, timeout=100)
    model = base
    if ablate_names:
        model = ablate(model, ablate_names)
    if edges:
        model = ablate_edges(model, edges)
    if rewire is not None:
        model = rewire(model)
    t0 = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    dt = time.time() - t0
    print(f"[{label:>45s}]  score = {score:7.3f} +/- {std:5.3f}  ({dt:.1f}s)")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("reflex", "all"):
        print("Ablation sweep for reflex slots 0..4.\n")
        run("baseline (no ablation)",        [])
        run("drop prox_left + prox_right",   ["prox_left", "prox_right"])
        run("drop safe_left + safe_right",   ["safe_left", "safe_right"])
        run("drop prox_left only",           ["prox_left"])
        run("drop prox_right only",          ["prox_right"])
        run("drop safe_left only",           ["safe_left"])
        run("drop safe_right only",          ["safe_right"])
        run("drop ALL four prox+safe",       ["prox_left", "prox_right", "safe_left", "safe_right"])

    if mode in ("corridor", "all"):
        print("\nAblation sweep for corridor / shortcut circuit.\n")
        # After collapsing the directional gate (COSBP/COSBN/NCR_E/NCR_W removed),
        # NEAR_CR = OR(NEAR_E, NEAR_W). Cutting either offset detector drops the
        # corresponding shortcut-firing opportunity.
        run("baseline (no ablation)", [])
        run("drop near_e (cut W[near_cr, near_e])",
            edges=[("near_cr", "near_e")])
        run("drop near_w (cut W[near_cr, near_w])",
            edges=[("near_cr", "near_w")])
        run("drop both near_e and near_w (kills shortcut)",
            edges=[("near_cr", "near_e"), ("near_cr", "near_w")])
