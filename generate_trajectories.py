"""Generate an offline dataset of synthetic 3D motion sequences."""

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "trajectories.npz"
DEFAULT_SEED = 42
DEFAULT_NOISE_STD = 0.05
OBSERVATION_STEPS = 50
FORECAST_STEPS = 25


def generate_trajectories(count, seed, noise_std, include_figure_eight=False):
    rng = np.random.default_rng(seed)
    labels = np.array(["straight", "mild_curve", "sharp_curve"])
    motion = rng.integers(len(labels), size=count)
    depth = rng.uniform(4.0, 8.0, count)
    position = np.column_stack((
        depth,
        depth * np.tan(rng.uniform(-1, 1, count) * np.deg2rad(60.0) / 6),
        depth * np.tan(rng.uniform(-1, 1, count) * np.deg2rad(47.0) / 6),
    ))
    yaw = rng.uniform(0, 2 * np.pi, count)
    speed = rng.uniform(7.0, 8.0, count)
    pitch = rng.uniform(-1, 1, count) * np.deg2rad(5.0)
    pitch_rate = rng.uniform(-0.1, 0.1, count)
    acceleration = rng.uniform(-0.5, 0.5, count)
    turn_sign = rng.choice([-1, 1], count)
    yaw_rate = np.zeros(count)
    min_rate = np.array([0.0, 0.01, 0.2])[motion]
    max_rate = np.array([0.0, 0.1, 1.5])[motion]
    steps = OBSERVATION_STEPS + FORECAST_STEPS
    trajectories = np.empty((count, steps, 3))
    dt = 0.2

    for step in range(steps):
        yaw_rate = np.clip(yaw_rate + rng.uniform(-0.02, 0.02, count), min_rate, max_rate)
        yaw += turn_sign * yaw_rate * dt
        pitch = np.clip(pitch + pitch_rate * dt, -np.deg2rad(80), np.deg2rad(80))
        speed = np.clip(speed + acceleration * dt, 4.0, 7.0)
        direction = np.column_stack((
            np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)
        ))
        position += speed[:, None] * direction * dt
        trajectories[:, step] = position

    kinds = labels[motion]
    if include_figure_eight:
        windows = []
        for spacing in (0.005, 0.01, 0.015, 0.02):
            phase = np.arange(0, 2 * np.pi, spacing)
            curve = np.column_stack((
                5 + 40 * np.sin(2 * phase) * np.cos(phase), 40 * np.sin(phase), np.zeros_like(phase)
            ))
            window_indices = (np.arange(len(curve))[:, None] + np.arange(steps)) % len(curve)
            windows.append(curve[window_indices])
        figure_eight = np.concatenate(windows)
        trajectories = np.concatenate((trajectories, figure_eight))
        kinds = np.concatenate((kinds, np.full(len(figure_eight), "figure_eight")))

    observations = trajectories[:, :OBSERVATION_STEPS].copy()
    if noise_std:
        observations += rng.normal(0, noise_std, size=observations.shape)
    return observations, trajectories[:, OBSERVATION_STEPS:], kinds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10_000, help="Number of independent trajectories.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42).")
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD,
                        help="Per-axis observation noise standard deviation in meters (default: 0.05).")
    parser.add_argument("--include-figure-eight", action="store_true", help="Add test-only windows.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.count < 1 or args.seed < 0 or not np.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("count must be positive; seed and finite noise-std must be nonnegative")

    observations, predictions, kinds = generate_trajectories(
        args.count, args.seed, args.noise_std, args.include_figure_eight
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as archive:
        np.savez_compressed(
            archive, observations=observations.astype(np.float32),
            predictions=predictions.astype(np.float32), kinds=kinds,
        )
    print(f"Saved {len(kinds):,} trajectories (seed={args.seed}, observation sigma={args.noise_std:g} m)"
          f" to {args.output.resolve()}")


if __name__ == "__main__":
    main()
