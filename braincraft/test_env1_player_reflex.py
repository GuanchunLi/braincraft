import sys
import unittest
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bot import Bot
from environment_1 import Environment
import env1_player_dummy2 as dummy2
import env1_player_dummy4 as dummy4
import env1_player_reflex as reflex


def rollout(model, seed, max_steps=None):
    Win, W, Wout, warmup, leak, f, g = model

    np.random.seed(seed)
    environment = Environment()
    bot = Bot()

    n_cam = bot.camera.resolution
    I = np.zeros((n_cam + 3, 1))
    X = np.zeros((1000, 1))

    bot.camera.update(bot.position, bot.direction, environment.world, environment.colormap)

    outputs = []
    true_dirs = [bot.direction]
    model_dirs = [np.pi / 2]
    true_pos = [np.array(bot.position, dtype=float)]
    model_pos = [np.array([0.5, 0.5])]
    reward_flags = [0.0]
    pulse_flags = [0.0]
    energies = [bot.energy]
    refill_steps = []
    startup_pulse = False
    hits = 0
    step = 0

    while bot.energy > 0 and (max_steps is None or step < max_steps):
        prev_energy = bot.energy
        I[:n_cam, 0] = 1 - bot.camera.depths
        I[n_cam:, 0] = bot.hit, bot.energy, 1.0
        X = (1 - leak) * X + leak * f(Win @ I + W @ X)
        O = float((Wout @ g(X)).item())
        outputs.append(O)

        if X[17, 0] > 0.5 and not refill_steps:
            startup_pulse = True

        if step > warmup:
            bot.forward(np.array([[O]]), environment, debug=False)
            hits += bot.hit
            if bot.energy > prev_energy:
                refill_steps.append(step + 1)

        model_dir = float(np.arctan2(X[9, 0], X[8, 0]))
        model_px = float(X[10, 0]) + 0.5
        model_py = float(X[11, 0]) + 0.5

        true_dirs.append(bot.direction)
        model_dirs.append(model_dir)
        true_pos.append(np.array(bot.position, dtype=float))
        model_pos.append(np.array([model_px, model_py]))
        reward_flags.append(float(X[18, 0]))
        pulse_flags.append(float(X[17, 0]))
        energies.append(bot.energy)

        step += 1

    true_dirs = np.array(true_dirs)
    model_dirs = np.array(model_dirs)
    true_pos = np.array(true_pos)
    model_pos = np.array(model_pos)
    reward_flags = np.array(reward_flags)
    pulse_flags = np.array(pulse_flags)
    energies = np.array(energies)

    dir_err = np.arctan2(np.sin(model_dirs - true_dirs), np.cos(model_dirs - true_dirs))
    pos_err = np.sqrt(np.sum((model_pos - true_pos) ** 2, axis=1))

    return {
        "outputs": np.array(outputs),
        "states": X,
        "steps": step,
        "hits": hits,
        "dir_err_deg": np.degrees(np.abs(dir_err)),
        "pos_err": pos_err,
        "reward_flags": reward_flags,
        "pulse_flags": pulse_flags,
        "energies": energies,
        "refill_steps": refill_steps,
        "startup_pulse": startup_pulse,
    }


class ReflexPlayerTests(unittest.TestCase):
    def test_make_activation_rejects_batched_input(self):
        _, _, _, _, _, f, _ = next(reflex.dummy_player())

        with self.assertRaisesRegex(ValueError, r"shape \(n,\) or \(n, 1\)"):
            f(np.zeros((1000, 2)))

    def test_readout_matches_dummy2_and_dummy4(self):
        seeds = [741858, 77285, 916765]
        reflex_model = next(reflex.dummy_player())
        dummy2_model = next(dummy2.dummy_player())
        dummy4_model = next(dummy4.dummy_player())

        for seed in seeds:
            with self.subTest(seed=seed):
                reflex_run = rollout(reflex_model, seed, max_steps=300)
                dummy2_run = rollout(dummy2_model, seed, max_steps=300)
                dummy4_run = rollout(dummy4_model, seed, max_steps=300)

                np.testing.assert_allclose(reflex_run["outputs"], dummy2_run["outputs"], atol=0.0, rtol=0.0)
                np.testing.assert_allclose(reflex_run["outputs"], dummy4_run["outputs"], atol=0.0, rtol=0.0)

    def test_tracker_rollout_stays_within_expected_error(self):
        seeds = [741858, 77285, 916765, 395393]
        reflex_model = next(reflex.dummy_player())

        dir_means = []
        pos_means = []
        dir_maxes = []
        pos_maxes = []
        for seed in seeds:
            run = rollout(reflex_model, seed)
            dir_means.append(float(run["dir_err_deg"].mean()))
            pos_means.append(float(run["pos_err"].mean()))
            dir_maxes.append(float(run["dir_err_deg"].max()))
            pos_maxes.append(float(run["pos_err"].max()))

        self.assertLess(np.mean(dir_means), 0.2)
        self.assertLess(max(dir_maxes), 5.1)
        self.assertLess(np.mean(pos_means), 0.02)
        self.assertLess(max(pos_maxes), 0.02)

    def test_reward_circuit_latches_without_startup_false_positive(self):
        seed = 741858
        reflex_model = next(reflex.dummy_player())
        run = rollout(reflex_model, seed)

        self.assertFalse(run["startup_pulse"])
        self.assertTrue(run["refill_steps"])

        first_refill = run["refill_steps"][0]
        first_pulse = first_refill + 1
        self.assertGreater(run["pulse_flags"][first_pulse], 0.5)
        self.assertGreater(run["reward_flags"][first_pulse + 1], 0.5)

        consecutive_positive = np.where(np.diff(run["energies"]) > 0)[0] + 1
        self.assertGreaterEqual(len(consecutive_positive), 2)

        first_block = consecutive_positive[:2]
        self.assertTrue(np.all(run["pulse_flags"][first_block + 1] > 0.5))


if __name__ == "__main__":
    unittest.main()
