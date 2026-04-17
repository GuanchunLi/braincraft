"""Detailed debug: inspect bio2 internal signals step by step."""
import numpy as np

from bot import Bot
from environment_2 import Environment
import env2_player_reflex_bio as bio1
import env2_player_reflex_bio2 as bio2


def _get_value(X, idx_map, key, default=None):
    if idx_map is None:
        return default
    idx = idx_map.get(key)
    if idx is None:
        return default
    return float(X[idx, 0])


def _format_optional(value, fmt):
    if value is None:
        return "n/a"
    return format(value, fmt)


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

        x10 = float(X_new[10, 0])
        x11 = float(X_new[11, 0])
        x19 = float(X_new[19, 0])
        x22 = float(X_new[22, 0])
        if bio2_idx is not None:
            cosn = _get_value(X_new, bio2_idx, "cos_n")
            cosbp = _get_value(X_new, bio2_idx, "cos_big_pos")
            cosbn = _get_value(X_new, bio2_idx, "cos_big_neg")
            near_e = _get_value(X_new, bio2_idx, "near_e")
            if near_e is None:
                near_e = _get_value(X_new, bio2_idx, "xe_pos")
            near_w = _get_value(X_new, bio2_idx, "near_w")
            if near_w is None:
                near_w = _get_value(X_new, bio2_idx, "xw_pos")
            ncre = _get_value(X_new, bio2_idx, "ncr_e")
            ncrw = _get_value(X_new, bio2_idx, "ncr_w")
            nc = _get_value(X_new, bio2_idx, "near_center")
            ncorr = _get_value(X_new, bio2_idx, "near_corr")
            hh = _get_value(X_new, bio2_idx, "heading_horiz")
            fc = _get_value(X_new, bio2_idx, "front_clear")
            es = _get_value(X_new, bio2_idx, "enough_steps")
            tsc = _get_value(X_new, bio2_idx, "trig_sc")
            ist = _get_value(X_new, bio2_idx, "is_turn")
            sc = _get_value(X_new, bio2_idx, "shortcut_steer", 0.0)
            init = _get_value(X_new, bio2_idx, "init_impulse", 0.0)
            x20 = sc + init
            print(f"{t:3d} X10={x10:+.3f} cosN={cosn:+.3f} CBP={cosbp:.1f} CBN={cosbn:.1f} "
                  f"NE={_format_optional(near_e, '.1f')} NW={_format_optional(near_w, '.1f')} "
                  f"NCE={ncre:.1f} NCW={ncrw:.1f} NCR={ncorr:.1f} "
                  f"HH={hh:.1f} FC={fc:.1f} ES={es:.1f} X22={x22:5.1f} "
                  f"TSC={tsc:.1f} IST={ist:.1f} "
                  f"SC={sc:+.3f} INIT={init:+.3f} SUM={x20:+.3f} "
                  f"O={O:+.3f} pos=({bot.position[0]:.3f},{bot.position[1]:.3f}) dir={np.degrees(bot.direction):.1f}")
        else:
            x20 = float(X_new[20, 0])
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
