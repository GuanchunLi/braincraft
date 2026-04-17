"""Debug: compare bio1 and bio2 step-by-step behavior."""
import numpy as np
import sys

from bot import Bot
from environment_2 import Environment
import env2_player_reflex_bio as bio1
import env2_player_reflex_bio2 as bio2


def trace_player(model_iter, label, n_steps=20, seed=12345, stride=1, use_eval_seed=True):
    # When use_eval_seed=True (default), mimic the evaluate() harness:
    # seed → 10 internal seeds → use seeds[0] for the environment.
    if use_eval_seed:
        np.random.seed(seed)
        seeds = np.random.randint(0, 1_000_000, 10)
        env_seed = int(seeds[0])
    else:
        env_seed = seed

    np.random.seed(seed)
    model = next(model_iter())
    W_in, W, W_out, warmup, leak, f, g = model

    np.random.seed(env_seed)
    environment = Environment()
    bot = Bot()
    p = bot.camera.resolution
    I, X = np.zeros((2*p+3, 1)), np.zeros((1000, 1))
    bot.camera.update(bot.position, bot.direction,
                      environment.world, environment.colormap)

    print(f"\n=== {label} ===")
    print(f"initial bot direction: {np.degrees(bot.direction):.3f} deg")

    total_dist = 0.0
    energy_log = []
    for t in range(n_steps):
        I[:p, 0] = 1 - bot.camera.depths
        I[p:2*p, 0] = bot.camera.values
        I[2*p:, 0] = bot.hit, bot.energy, 1.0

        pre = W_in @ I + W @ X
        X_new = (1-leak)*X + leak*f(pre)
        O = (W_out @ g(X_new)).item()

        if bot.energy <= 0 and t > 0:
            break
        if t % stride == 0 or t == n_steps - 1:
            print(f"t={t:4d}  pos=({bot.position[0]:.3f},{bot.position[1]:.3f})  dir={np.degrees(bot.direction):6.2f}  "
                  f"O={O:+.4f}  hit={int(bot.hit)}  E={bot.energy:.4f}  dist={total_dist:.3f}")

        X = X_new
        prev_pos = np.array(bot.position, dtype=float).copy()
        bot.forward(np.array([[O]]), environment)
        total_dist += float(np.linalg.norm(np.array(bot.position) - prev_pos))
    print(f"END   pos=({bot.position[0]:.3f},{bot.position[1]:.3f})  E={bot.energy:.4f}  total_dist={total_dist:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    trace_player(bio1.reflex_bio_player, "BIO1", n_steps=a.steps, stride=a.stride)
    trace_player(bio2.reflex_bio2_player, "BIO2", n_steps=a.steps, stride=a.stride)
