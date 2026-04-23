"""Parallel 500-seed validation of env1_player_bio.py.

Each outer seed reproduces the evaluate() harness exactly:
    np.random.seed(outer_seed)
    seeds = np.random.randint(0, 1_000_000, 10)
    -> 10 inner episodes with those seeds.

Reports per-seed mean and std, plus aggregate stats across seeds.
Use under the `braincraft` conda env (numpy 2.x).
"""

from __future__ import annotations

import os

# Pin BLAS to a single thread per worker — multi-threaded BLAS × many
# worker processes thrashes the kernel scheduler. Must be set BEFORE
# numpy is imported (here and in spawned workers, which inherit env).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np

# Module-level worker state — built once per worker process.
_MODEL = None
_BOT_CLS = None
_ENV_CLS = None


def _init_worker():
    global _MODEL, _BOT_CLS, _ENV_CLS
    import env1_player_bio as bio
    from bot import Bot
    from environment_1 import Environment

    _MODEL = next(bio.bio_player())
    _BOT_CLS = Bot
    _ENV_CLS = Environment


def _eval_one(outer_seed: int):
    """Mirror challenge_1.evaluate() but inline so we can return per-run scores."""
    W_in, W, W_out, warmup, leak, f, g = _MODEL

    np.random.seed(outer_seed)
    inner_seeds = np.random.randint(0, 1_000_000, 10)

    scores = np.empty(10, dtype=np.float64)
    for i, s in enumerate(inner_seeds):
        np.random.seed(int(s))
        env = _ENV_CLS()
        bot = _BOT_CLS()

        n = bot.camera.resolution
        I = np.zeros((n + 3, 1))
        X = np.zeros((1000, 1))

        bot.camera.update(bot.position, bot.direction, env.world, env.colormap)

        distance = 0.0
        iteration = 0
        while bot.energy > 0:
            I[:n, 0] = 1 - bot.camera.depths
            I[n:, 0] = bot.hit, bot.energy, 1.0

            X = (1 - leak) * X + leak * f(W_in @ I + W @ X)
            O = W_out @ g(X)

            if iteration > warmup:
                p = bot.position
                bot.forward(O, env, False)
                distance += float(np.linalg.norm(p - bot.position))
            iteration += 1
        scores[i] = distance

    return outer_seed, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--start", type=int, default=0,
                    help="First outer seed (sequential range [start, start+n_seeds)).")
    ap.add_argument("--out", type=str, default="validate_env1_player_bio.npz")
    ap.add_argument("--report", action="store_true",
                    help="Print per-seed lines (verbose).")
    args = ap.parse_args()

    seeds = list(range(args.start, args.start + args.n_seeds))

    print(f"numpy = {np.__version__}")
    print(f"workers = {args.workers}, seeds = {len(seeds)} "
          f"(range {seeds[0]}..{seeds[-1]}), 10 runs each "
          f"= {len(seeds) * 10} episodes")

    t0 = time.time()
    all_scores = np.empty((len(seeds), 10), dtype=np.float64)
    done = 0

    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for outer_seed, scores in pool.imap_unordered(_eval_one, seeds, chunksize=1):
            row = outer_seed - args.start
            all_scores[row] = scores
            done += 1
            if args.report or done % 25 == 0 or done == len(seeds):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(seeds) - done) / max(rate, 1e-6)
                print(f"  [{done:4d}/{len(seeds)}]  seed={outer_seed:5d}  "
                      f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
                      f"min={scores.min():.4f}  "
                      f"({elapsed:6.1f}s elapsed, ETA {eta:5.1f}s)",
                      flush=True)

    elapsed = time.time() - t0
    seed_means = all_scores.mean(axis=1)
    seed_stds = all_scores.std(axis=1)

    print("")
    print("=" * 66)
    print(f"Completed {len(seeds)} outer seeds in {elapsed:.1f}s "
          f"({len(seeds) * 10 / elapsed:.1f} episodes/s)")
    print("=" * 66)
    print(f"Across-seed-mean  mean = {seed_means.mean():.4f}")
    print(f"Across-seed-mean  std  = {seed_means.std():.4f}")
    print(f"Across-seed-mean  min  = {seed_means.min():.4f}")
    print(f"Across-seed-mean  max  = {seed_means.max():.4f}")
    print("")
    print(f"Within-seed std   mean = {seed_stds.mean():.4f}")
    print(f"Within-seed std   max  = {seed_stds.max():.4f}")
    print("")

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    qvals = np.quantile(seed_means, qs)
    print("Per-seed-mean quantiles:")
    for q, v in zip(qs, qvals):
        print(f"  q={q:.2f}  {v:.4f}")
    print("")

    thresholds = [12.0, 13.0, 13.5, 14.0, 14.5, 14.6, 14.7]
    print("Catastrophic-failure tail (per-seed mean below threshold):")
    for thr in thresholds:
        n_below = int(np.sum(seed_means < thr))
        print(f"  seeds with mean <  {thr:5.2f}: {n_below:4d}/{len(seeds)} "
              f"({100.0 * n_below / len(seeds):5.2f}%)")
    print("")

    ep_thresholds = [12.0, 13.0, 14.0, 14.5]
    flat = all_scores.ravel()
    print("Per-episode tail (across all 5000 episodes):")
    for thr in ep_thresholds:
        n_below = int(np.sum(flat < thr))
        print(f"  episodes <  {thr:5.2f}: {n_below:5d}/{len(flat)} "
              f"({100.0 * n_below / len(flat):5.2f}%)")
    print("")

    if seed_means.size > 0:
        worst_order = np.argsort(seed_means)[:10]
        print("Worst 10 seeds (by per-seed mean):")
        for r in worst_order:
            outer_seed = args.start + int(r)
            print(f"  seed={outer_seed:5d}  mean={seed_means[r]:.4f}  "
                  f"std={seed_stds[r]:.4f}  min={all_scores[r].min():.4f}")

    np.savez_compressed(
        args.out,
        seeds=np.asarray(seeds, dtype=np.int64),
        scores=all_scores,
        numpy_version=np.__version__,
    )
    print(f"\nSaved per-episode scores to {args.out}")


if __name__ == "__main__":
    main()
