"""Visualize trajectories of one or more env1 players on environment_1.

Replays the same RNN loop as challenge_1.evaluate() but records the bot
position at every step and overlays it on the world map. Supports plotting
multiple models side-by-side (rows) across multiple seeds (columns).
"""

import argparse
import importlib
import inspect

import numpy as np
if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2  # compat shim for older NumPy used by camera.py
import matplotlib.pyplot as plt

from bot import Bot
from environment_1 import Environment


# Registry of known env1 players: label -> (module name, generator fn name).
# Add new entries here to expose them on the command line.
# Default models to plot when no --models flag is passed on the CLI.
# Edit this list to change which models render by default.
DEFAULT_MODELS = ["ensemble_v3_metric"]

# Default seeds (one column per seed).
DEFAULT_SEEDS = [12345, 1, 2, 7]


PLAYERS = {
    "dummy":              ("env1_player_dummy",              "dummy_player"),
    "random":             ("env1_player_random",             "random_player"),
    "simple":             ("env1_player_simple",             "simple_player"),
    "switcher":           ("env1_player_switcher",           "switcher_player"),
    "switcher_alt":       ("env1_player_switcher_alt",       "switcher_player"),
    "evolution":          ("env1_player_evolution",          "evolutionary_player"),
    "ensemble_v3_metric": ("env1_player_ensemble_v3_metric", "ensemble_v3_metric_env1_player"),
}


def load_model(player_key):
    """Import a player module and pull the first model out of its generator."""
    if player_key not in PLAYERS:
        raise KeyError(f"unknown player '{player_key}'. "
                       f"Known: {sorted(PLAYERS)}")
    module_name, fn_name = PLAYERS[player_key]
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    gen = fn() if inspect.isfunction(fn) or inspect.ismethod(fn) else fn
    model = next(iter(gen))
    # Some generators yield (model, meta); normalize to a 7-tuple.
    if isinstance(model, tuple) and len(model) == 2 and isinstance(model[0], tuple):
        model = model[0]
    return model


def run_trajectory(player_key, seed=12345, max_steps=None):
    np.random.seed(seed)
    environment = Environment()
    bot = Bot()

    W_in, W, W_out, warmup, leak, f, g = load_model(player_key)

    n = bot.camera.resolution
    I = np.zeros((n + 3, 1))
    X = np.zeros((W.shape[0], 1))

    bot.camera.update(bot.position, bot.direction,
                      environment.world, environment.colormap)

    positions = [bot.position]
    directions = [bot.direction]
    outputs = []
    hits_steps = []
    iteration = 0
    distance = 0.0

    while bot.energy > 0:
        I[:n, 0] = 1 - bot.camera.depths
        I[n:, 0] = bot.hit, bot.energy, 1.0
        X = (1 - leak) * X + leak * f(np.dot(W_in, I) + np.dot(W, X))
        O = np.dot(W_out, g(X))

        if iteration > warmup:
            p = bot.position
            bot.forward(O, environment, debug=False)
            distance += np.linalg.norm(np.array(p) - np.array(bot.position))
            if bot.hit:
                hits_steps.append(len(positions))
        positions.append(bot.position)
        directions.append(bot.direction)
        outputs.append(float(np.asarray(O).flatten()[0]))
        iteration += 1
        if max_steps is not None and iteration >= max_steps:
            break

    return {
        "positions": np.array(positions),
        "directions": np.array(directions),
        "outputs": np.array(outputs),
        "hits": np.array(hits_steps, dtype=int),
        "distance": distance,
        "world_rgb": environment.world_rgb,
        "world": environment.world,
    }


def plot_trajectory(traj, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    world = traj["world"]
    extent = [0, world.shape[1] / max(world.shape),
              0, world.shape[0] / max(world.shape)]
    ax.imshow(traj["world_rgb"], origin="lower", extent=extent,
              interpolation="nearest")
    P = traj["positions"]
    t = np.linspace(0, 1, len(P))
    ax.scatter(P[:, 0], P[:, 1], c=t, cmap="viridis", s=2, zorder=10)
    ax.plot(P[0, 0], P[0, 1], "o", color="lime", markersize=10,
            markeredgecolor="black", label="start", zorder=20)
    ax.plot(P[-1, 0], P[-1, 1], "X", color="red", markersize=10,
            markeredgecolor="black", label="end", zorder=20)
    if len(traj["hits"]):
        H = P[traj["hits"]]
        ax.scatter(H[:, 0], H[:, 1], facecolor="none", edgecolor="red",
                   s=40, linewidth=1.0, label="hit", zorder=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_models(player_keys, seeds, max_steps=None, save_path=None, show=True):
    """Render a grid: one row per model, one column per seed."""
    n_rows = len(player_keys)
    n_cols = len(seeds)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for i, key in enumerate(player_keys):
        for j, seed in enumerate(seeds):
            try:
                traj = run_trajectory(key, seed=seed, max_steps=max_steps)
                plot_trajectory(
                    traj, ax=axes[i][j],
                    title=f"{key}  seed={seed}\n"
                          f"dist={traj['distance']:.2f}  "
                          f"hits={len(traj['hits'])}",
                )
                print(f"{key:>20s}  seed={seed}: "
                      f"distance={traj['distance']:.3f}, "
                      f"steps={len(traj['positions'])}, "
                      f"hits={len(traj['hits'])}")
            except Exception as e:
                axes[i][j].set_title(f"{key} seed={seed}\nFAILED: {e}",
                                     fontsize=8, color="red")
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                print(f"{key} seed={seed}: FAILED -> {e}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    if show:
        plt.show()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot env1 trajectories for one or more player models.")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Player keys to plot. Choices: {sorted(PLAYERS)} "
             f"(or 'all' to plot every registered player). "
             f"Default: {DEFAULT_MODELS}")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Random seeds (one column per seed).")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save", default=None,
                        help="Output image path. Default: "
                             "trajectory_<model1>[_<model2>...].png")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = sorted(PLAYERS) if args.models == ["all"] else args.models
    save_path = args.save or f"trajectory_env1_{'_'.join(models)}.png"
    plot_models(
        player_keys=models,
        seeds=args.seeds,
        max_steps=args.max_steps,
        save_path=save_path,
        show=not args.no_show,
    )
