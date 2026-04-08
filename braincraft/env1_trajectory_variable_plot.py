"""Visualize an env1 player trajectory together with a per-step variable.

Like env1_trajectory_plot.py, but in addition to drawing the bot's path on
the world map it records the value of a user-specified expression at every
step and plots it as a time series next to the map. The trajectory dots are
also recolored by that variable, so spatial and temporal structure can be
read together.

Examples
--------
    python env1_trajectory_variable_plot.py
    python env1_trajectory_variable_plot.py --var "prox[11]"
    python env1_trajectory_variable_plot.py --var "X[8,0]" --var-label is_rewarded
    python env1_trajectory_variable_plot.py --models dummy3 --seeds 12345 1
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

from env1_trajectory_plot import PLAYERS, load_model


DEFAULT_MODELS = ["dummy3"]
DEFAULT_SEEDS = [12345, 1, 2, 7]
DEFAULT_VAR = "prox[52]-prox[11]"  # prox[left_side_idx] in dummy3


def run_trajectory_with_var(player_key, var_expr, seed=12345, max_steps=None):
    """Replay the player and record `var_expr` each step.

    The expression is eval'd in a namespace exposing:
        prox  -- 1 - bot.camera.depths  (length n)
        I     -- raw input vector ((n+3, 1))
        X     -- reservoir state ((N, 1))
        O     -- output vector
        bot   -- the Bot instance
        np    -- numpy
    """
    np.random.seed(seed)
    environment = Environment()
    bot = Bot()

    W_in, W, W_out, warmup, leak, f, g = load_model(player_key)

    n = bot.camera.resolution
    I = np.zeros((n + 3, 1))
    X = np.zeros((W.shape[0], 1))

    bot.camera.update(bot.position, bot.direction,
                      environment.world, environment.colormap)

    code = compile(var_expr, "<var>", "eval")

    def eval_var(O):
        ns = {
            "prox": I[:n, 0],
            "I": I,
            "X": X,
            "O": O,
            "bot": bot,
            "np": np,
        }
        return float(np.asarray(eval(code, ns)).reshape(-1)[0])

    positions = [bot.position]
    var_values = []
    hits_steps = []
    iteration = 0
    distance = 0.0

    while bot.energy > 0:
        I[:n, 0] = 1 - bot.camera.depths
        I[n:, 0] = bot.hit, bot.energy, 1.0
        X = (1 - leak) * X + leak * f(np.dot(W_in, I) + np.dot(W, X))
        O = np.dot(W_out, g(X))

        var_values.append(eval_var(O))

        if iteration > warmup:
            p = bot.position
            bot.forward(O, environment, debug=False)
            distance += np.linalg.norm(np.array(p) - np.array(bot.position))
            if bot.hit:
                hits_steps.append(len(positions))
        positions.append(bot.position)
        iteration += 1
        if max_steps is not None and iteration >= max_steps:
            break

    # positions has one more entry than var_values (initial pose). Align by
    # dropping the initial pose for the per-step coloring/series.
    return {
        "positions": np.array(positions),
        "var": np.array(var_values),
        "hits": np.array(hits_steps, dtype=int),
        "distance": distance,
        "world_rgb": environment.world_rgb,
        "world": environment.world,
    }


def plot_trajectory_with_var(traj, var_label, ax_map, ax_ts, title=None):
    world = traj["world"]
    extent = [0, world.shape[1] / max(world.shape),
              0, world.shape[0] / max(world.shape)]
    ax_map.imshow(traj["world_rgb"], origin="lower", extent=extent,
                  interpolation="nearest")

    P = traj["positions"]
    V = traj["var"]
    # Color the per-step positions by the variable. Skip the initial pose so
    # P_step and V align.
    P_step = P[1:1 + len(V)]
    sc = ax_map.scatter(P_step[:, 0], P_step[:, 1], c=V, cmap="coolwarm",
                        s=4, zorder=10)
    ax_map.plot(P[0, 0], P[0, 1], "o", color="lime", markersize=10,
                markeredgecolor="black", label="start", zorder=20)
    ax_map.plot(P[-1, 0], P[-1, 1], "X", color="red", markersize=10,
                markeredgecolor="black", label="end", zorder=20)
    if len(traj["hits"]):
        H = P[traj["hits"]]
        ax_map.scatter(H[:, 0], H[:, 1], facecolor="none", edgecolor="red",
                       s=40, linewidth=1.0, label="hit", zorder=15)
    ax_map.set_xlim(0, 1)
    ax_map.set_ylim(0, 1)
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper right", fontsize=7)
    if title:
        ax_map.set_title(title, fontsize=9)

    cbar = plt.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label(var_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Time series
    ax_ts.plot(V, color="C0", linewidth=1.0)
    if len(traj["hits"]):
        for h in traj["hits"]:
            ax_ts.axvline(h - 1, color="red", alpha=0.3, linewidth=0.6)
    ax_ts.set_xlabel("step", fontsize=8)
    ax_ts.set_ylabel(var_label, fontsize=8)
    ax_ts.tick_params(labelsize=7)
    ax_ts.grid(True, alpha=0.3)


def plot_models(player_keys, seeds, var_expr, var_label,
                max_steps=None, save_path=None, show=True):
    n_rows = len(player_keys) * len(seeds)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(11, 4.5 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.2]},
    )

    row = 0
    for key in player_keys:
        for seed in seeds:
            ax_map, ax_ts = axes[row]
            try:
                traj = run_trajectory_with_var(
                    key, var_expr, seed=seed, max_steps=max_steps)
                plot_trajectory_with_var(
                    traj, var_label,
                    ax_map=ax_map, ax_ts=ax_ts,
                    title=f"{key}  seed={seed}\n"
                          f"dist={traj['distance']:.2f}  "
                          f"hits={len(traj['hits'])}",
                )
                print(f"{key:>20s}  seed={seed}: "
                      f"distance={traj['distance']:.3f}, "
                      f"steps={len(traj['positions'])}, "
                      f"hits={len(traj['hits'])}")
            except Exception as e:
                ax_map.set_title(f"{key} seed={seed}\nFAILED: {e}",
                                 fontsize=8, color="red")
                ax_map.set_xticks([]); ax_map.set_yticks([])
                ax_ts.set_xticks([]); ax_ts.set_yticks([])
                print(f"{key} seed={seed}: FAILED -> {e}")
            row += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    if show:
        plt.show()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot env1 trajectories alongside a per-step variable.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Player keys. Choices: {sorted(PLAYERS)}")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--var", default=DEFAULT_VAR,
                        help="Python expression evaluated each step. "
                             "Namespace: prox, I, X, O, bot, np. "
                             f"Default: {DEFAULT_VAR!r}")
    parser.add_argument("--var-label", default=None,
                        help="Label shown on the colorbar / y-axis. "
                             "Defaults to the expression.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = sorted(PLAYERS) if args.models == ["all"] else args.models
    label = args.var_label or args.var
    safe_var = "".join(c if c.isalnum() else "_" for c in args.var).strip("_")
    save_path = (args.save or
                 f"trajectory_var_{safe_var}_{'_'.join(models)}.png")
    plot_models(
        player_keys=models,
        seeds=args.seeds,
        var_expr=args.var,
        var_label=label,
        max_steps=args.max_steps,
        save_path=save_path,
        show=not args.no_show,
    )
