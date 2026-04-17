"""Detailed debug: inspect reflex features X0-X5 at each step."""
import numpy as np

from bot import Bot
from environment_2 import Environment
import env2_player_reflex_bio as bio1
import env2_player_reflex_bio2 as bio2


def trace_player(model_iter, label, n_steps=20, seed=12345, bio2_idx=None):
    np.random.seed(seed)
    seeds = np.random.randint(0, 1_000_000, 10)
    env_seed = int(seeds[0])

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
    for t in range(n_steps):
        I[:p, 0] = 1 - bot.camera.depths
        I[p:2*p, 0] = bot.camera.values
        I[2*p:, 0] = bot.hit, bot.energy, 1.0

        pre = W_in @ I + W @ X
        X_new = (1-leak)*X + leak*f(pre)
        O = (W_out @ g(X_new)).item()

        x5 = float(X_new[5, 0])
        x10 = float(X_new[10, 0])
        x11 = float(X_new[11, 0])
        x18 = float(X_new[18, 0])
        x19 = float(X_new[19, 0])
        x20 = float(X_new[20, 0])
        x22 = float(X_new[22, 0])
        if bio2_idx is not None:
            cosn = float(X_new[bio2_idx["cos_n"], 0])
            cosbp = float(X_new[bio2_idx["cos_big_pos"], 0])
            cosbn = float(X_new[bio2_idx["cos_big_neg"], 0])
            xep = float(X_new[bio2_idx["xe_pos"], 0])
            xen = float(X_new[bio2_idx["xe_neg"], 0])
            ncre = float(X_new[bio2_idx["ncr_e"], 0])
            nc = float(X_new[bio2_idx["near_center"], 0])
            ncorr = float(X_new[bio2_idx["near_corr"], 0])
            hh = float(X_new[bio2_idx["heading_horiz"], 0])
            fc = float(X_new[bio2_idx["front_clear"], 0])
            es = float(X_new[bio2_idx["enough_steps"], 0])
            tsc = float(X_new[bio2_idx["trig_sc"], 0])
            ist = float(X_new[bio2_idx["is_turn"], 0])
            ttp = float(X_new[bio2_idx["tt_plus"], 0])
            ttm = float(X_new[bio2_idx["tt_minus"], 0])
            print(f"{t:3d} X10={x10:+.3f} cosN={cosn:+.3f} CBP={cosbp:.1f} CBN={cosbn:.1f} "
                  f"XEP={xep:.1f} XEN={xen:.1f} NCE={ncre:.1f} NCR={ncorr:.1f} "
                  f"HH={hh:.1f} FC={fc:.1f} ES={es:.1f} X22={x22:5.1f} "
                  f"TSC={tsc:.1f} IST={ist:.1f} TTP={ttp:.2f} TTM={ttm:.2f} "
                  f"X20={x20:+.3f} O={O:+.3f} pos=({bot.position[0]:.3f},{bot.position[1]:.3f}) dir={np.degrees(bot.direction):.1f}")
        else:
            print(f"{t:3d} X10={x10:+.3f} X11={x11:+.3f} X19={x19:6.1f} X22={x22:5.1f} "
                  f"X20={x20:+.3f} O={O:+.3f} pos=({bot.position[0]:.3f},{bot.position[1]:.3f}) dir={np.degrees(bot.direction):.1f}")

        X = X_new
        bot.forward(np.array([[O]]), environment)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15)
    a = ap.parse_args()
    trace_player(bio1.reflex_bio_player, "BIO1", n_steps=a.steps)
    bio2_idx = bio2._bio2_indices(64)
    trace_player(bio2.reflex_bio2_player, "BIO2", n_steps=a.steps, bio2_idx=bio2_idx)
