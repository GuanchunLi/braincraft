"""Visualize the trajectory of env1_player_dummy on environment_1.

Replays the same RNN loop as challenge_1.evaluate() but records the bot
position at every step and overlays it on the world map.
"""

import numpy as np
if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2  # compat shim for older NumPy used by camera.py
import matplotlib.pyplot as plt

from bot import Bot
from environment_1 import Environment
from env1_player_dummy import dummy_player


def run_trajectory(seed=12345, max_steps=None):
    np.random.seed(seed)
    environment = Environment()
    bot = Bot()

    model = next(dummy_player())
    W_in, W, W_out, warmup, leak, f, g = model

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
        outputs.append(float(O.item()))
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
    # color trajectory by time
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
    ax.legend(loc="upper right", fontsize=8)
    if title:
        ax.set_title(title)
    return ax


if __name__ == "__main__":
    seeds = [12345, 1, 2, 7]
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 5))
    for ax, s in zip(axes, seeds):
        traj = run_trajectory(seed=s)
        plot_trajectory(
            traj, ax=ax,
            title=f"seed={s}  dist={traj['distance']:.2f}  "
                  f"hits={len(traj['hits'])}",
        )
        print(f"seed={s}: distance={traj['distance']:.3f}, "
              f"steps={len(traj['positions'])}, hits={len(traj['hits'])}")
    plt.tight_layout()
    plt.savefig("trajectory_dummy.png", dpi=120)
    print("Saved trajectory_dummy.png")
    plt.show()
