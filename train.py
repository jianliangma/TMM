"""Train TMM-F and/or TMM-N and compare them with CV and CA on synthetic trajectories."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from numba import get_num_threads, set_num_threads
from threadpoolctl import threadpool_limits

from generate_trajectories import DEFAULT_DATA, DEFAULT_SEED, FORECAST_STEPS, OBSERVATION_STEPS, ROOT
from geometry import (
    cross_distances,
    kinematic_forecasts,
    nearest_neighbor_references,
    normalize_trajectories,
    training_distances,
    trajectory_errors,
)
from report import save_comparison, save_plots
from tmm import MDN, NUM_MIXTURES, fit_coefficients, mdn_loss, rbf_basis, sample_modes


METRICS = ("ADE", "ED", "DF")
VARIANTS = {"f": "TMM-F", "n": "TMM-N"}


def load_trajectories(path):
    with np.load(path, allow_pickle=False) as archive:
        observations = archive["observations"].astype(np.float64)
        futures = archive["predictions"].astype(np.float64)
        kinds = archive["kinds"]
    if kinds.ndim != 1 or observations.shape != (len(kinds), OBSERVATION_STEPS, 3):
        raise ValueError(f"expected observations shaped (N, {OBSERVATION_STEPS}, 3) and N motion labels")
    if futures.shape != (len(kinds), FORECAST_STEPS, 3):
        raise ValueError(f"expected predictions shaped (N, {FORECAST_STEPS}, 3)")
    if not np.isfinite(observations).all() or not np.isfinite(futures).all():
        raise ValueError("trajectory coordinates must be finite")
    return observations, futures, kinds


def make_splits(kinds, seed, limit=None):
    independent = np.flatnonzero(kinds != "figure_eight")
    shuffled = np.random.default_rng(seed).permutation(independent)[:limit]
    if len(shuffled) < 10:
        raise ValueError("at least 10 independent trajectories are needed for train/val/test")
    train_end = int(0.7 * len(shuffled))
    val_end = train_end + int(0.1 * len(shuffled))
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:], np.flatnonzero(kinds == "figure_eight")


def make_features(observations, train_ids, reference_count):
    distances = training_distances(observations[train_ids])
    selected = nearest_neighbor_references(distances, reference_count)
    train_features = np.exp(-0.005 * distances[:, selected] ** 2).astype(np.float32)
    return observations[train_ids[selected]], train_features


def kernel_features(observations, references):
    return np.exp(-0.005 * cross_distances(observations, references) ** 2).astype(np.float32)


@torch.inference_mode()
def infer(model, features):
    model.eval()
    return torch.cat([model(batch) for batch in features.split(256)])


def validation_frechet(parameters, obs, truth, transforms, basis, case_ids, seed, samples):
    rng = np.random.default_rng(seed)
    distances = []
    for index in case_ids:
        transform = () if transforms is None else (transforms[0][index], transforms[1][index])
        modes, weights = sample_modes(parameters[index], basis, obs[index, -1], rng, samples, *transform)
        distances.append(weights @ [trajectory_errors(mode, truth[index])[2] for mode in modes])
    return float(np.mean(distances))


def fit_model(train_x, train_y, val_x, val_y, val_obs, val_future, val_transforms, basis, args):
    model = MDN(train_x.shape[1]).to(train_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    validation_ids = np.random.default_rng(args.seed).permutation(len(val_x))[:args.val_cases]
    history = []
    best_score = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), device=train_x.device)
        total_loss = torch.zeros((), device=train_x.device)
        for indices in permutation.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = mdn_loss(train_y[indices], model(train_x[indices]))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(indices)

        val_parameters = infer(model, val_x)
        train_loss = total_loss.item() / len(train_x)
        val_loss = mdn_loss(val_y, val_parameters).item()
        val_distance = validation_frechet(
            val_parameters.cpu().numpy(), val_obs, val_future, val_transforms,
            basis, validation_ids, args.seed, args.val_samples,
        )
        if not np.isfinite([train_loss, val_loss, val_distance]).all():
            raise RuntimeError("Training produced a non-finite validation or training metric.")
        if val_distance < best_score:
            best_score = val_distance
            best_epoch = epoch
            best_state = {name: value.detach().to(device="cpu", copy=True) for name, value in model.state_dict().items()}
        history.append(dict(epoch=epoch, train_nll=train_loss, val_nll=val_loss, val_mean_df=val_distance))
        print(f"Epoch {epoch:3}/{args.epochs} | train NLL {train_loss:8.3f} | val NLL {val_loss:8.3f}"
              f" | val M-DF {val_distance:6.3f} m", flush=True)

    model.load_state_dict(best_state)
    return model, history, best_epoch


def average_metrics(scores):
    return {name: dict(zip(METRICS, np.mean(values, axis=0).tolist())) for name, values in scores.items()}


def evaluate_baselines(obs, truth):
    scores = {"CV": [], "CA": []}
    for observation, future in zip(obs, truth):
        for name, prediction in zip(scores, kinematic_forecasts(observation, len(future))):
            scores[name].append(trajectory_errors(prediction, future))
    return average_metrics(scores)


def evaluate(parameters, obs, truth, transforms, basis, kinds, ids, variant, args):
    rng = np.random.default_rng(args.seed + 1)
    top1, weighted, best = (f"{variant} {name}" for name in ("top-1", "weighted", "oracle"))
    scores = {name: [] for name in (top1, weighted, best)}
    sharp_ids = np.flatnonzero(kinds == "sharp_curve")
    preview_index = sharp_ids[0] if len(sharp_ids) else 0
    preview = None
    for index, distribution in enumerate(parameters):
        observation = obs[index]
        future = truth[index]
        transform = () if transforms is None else (transforms[0][index], transforms[1][index])
        modes, weights = sample_modes(
            distribution, basis, observation[-1], rng, args.samples, *transform
        )
        mode_scores = np.stack([trajectory_errors(mode, future) for mode in modes])
        scores[top1].append(mode_scores[0])
        scores[weighted].append(weights @ mode_scores)
        scores[best].append(mode_scores.min(axis=0))
        if index == preview_index:
            cv, ca = kinematic_forecasts(observation, len(basis))
            preview = (observation, future, modes, weights, cv, ca, str(kinds[index]), int(ids[index]))

    return average_metrics(scores), preview


def print_metrics(metrics, name, size):
    print(f"\n{name} ({size:,} trajectories, meters)")
    print(f"{'Method':<16} {'ADE':>9} {'ED / FDE':>9} {'DF':>9}")
    for method, scores in metrics.items():
        print(f"{method:<16} {scores['ADE']:9.4f} {scores['ED']:9.4f} {scores['DF']:9.4f}")


def run_variant(args, device, variant, obs, truth, kinds, splits):
    started = perf_counter()
    torch.manual_seed(args.seed)
    train_ids, val_ids, test_ids, figure_ids = splits
    model_obs, model_truth = obs, truth
    transforms = None
    if variant == "TMM-N":
        model_obs, model_truth, rotations, scales = normalize_trajectories(obs, truth)
        transforms = (rotations, scales)

    def transforms_for(ids):
        return None if transforms is None else (transforms[0][ids], transforms[1][ids])

    print(f"\n{variant}: computing training Frechet distances and selecting representatives...", flush=True)
    references, train_features = make_features(model_obs, train_ids, args.references)
    val_features = kernel_features(model_obs[val_ids], references)
    fit_ids = np.concatenate((train_ids, val_ids))
    coefficients = fit_coefficients(model_obs[fit_ids], model_truth[fit_ids])
    basis = rbf_basis(np.arange(FORECAST_STEPS + 1))
    basis = basis[1:] - basis[:1]

    def tensor(array):
        return torch.from_numpy(array).to(device)

    print(f"Training with {len(references)} references; preprocessing took {perf_counter() - started:.1f}s.", flush=True)
    model, history, best_epoch = fit_model(
        tensor(train_features), tensor(coefficients[:len(train_ids)]), tensor(val_features),
        tensor(coefficients[len(train_ids):]), obs[val_ids], truth[val_ids], transforms_for(val_ids), basis, args,
    )

    def score(ids):
        features = kernel_features(model_obs[ids], references)
        parameters = infer(model, tensor(features)).cpu().numpy()
        return evaluate(parameters, obs[ids], truth[ids], transforms_for(ids), basis, kinds[ids], ids, variant, args)

    print(f"\nEvaluating {variant} at its best validation epoch ({best_epoch})...", flush=True)
    test_metrics, preview = score(test_ids)
    print_metrics(test_metrics, f"{variant} held-out test", len(test_ids))
    results = {"references": len(references), "best_epoch": best_epoch, "test": test_metrics}
    if len(figure_ids):
        figure_metrics, _ = score(figure_ids)
        print_metrics(figure_metrics, f"{variant} figure-eight holdout", len(figure_ids))
        results["figure_eight"] = figure_metrics

    directory = args.output_dir / variant.lower()
    directory.mkdir(parents=True, exist_ok=True)
    save_plots(directory, history, preview, variant)
    results["elapsed_seconds"] = round(perf_counter() - started, 2)
    with (directory / "history.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return results


def run(args, device):
    started = perf_counter()
    obs, truth, kinds = load_trajectories(args.data)
    splits = make_splits(kinds, args.seed, args.max_trajectories)
    train_ids, val_ids, test_ids, figure_ids = splits
    print(f"Device: {device} | seed: {args.seed} | train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}",
          flush=True)
    variants = list(VARIANTS.values()) if args.variant == "both" else [VARIANTS[args.variant]]
    results = {
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "split_sizes": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "forecast_steps": FORECAST_STEPS, "unit": "m", "variants": {}, "test": {},
    }
    evaluation_sets = {"test": test_ids}
    if len(figure_ids):
        results["split_sizes"]["figure_eight"] = len(figure_ids)
        results["figure_eight"] = {}
        evaluation_sets["figure_eight"] = figure_ids

    for variant in variants:
        variant_results = run_variant(args, device, variant, obs, truth, kinds, splits)
        results["variants"][variant] = {key: variant_results[key] for key in ("references", "best_epoch", "elapsed_seconds")}
        for key in evaluation_sets:
            results[key].update(variant_results[key])

    for key, ids in evaluation_sets.items():
        results[key].update(evaluate_baselines(obs[ids], truth[ids]))
        print_metrics(results[key], "Held-out test" if key == "test" else "Figure-eight holdout", len(ids))
    save_comparison(args.output_dir, results["test"], variants, len(test_ids))
    results["elapsed_seconds"] = round(perf_counter() - started, 2)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"\nFinished in {results['elapsed_seconds']:.1f}s. Results: {args.output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--variant", choices=("both", "f", "n"), default="both",
                        help="Train both independently, only raw-coordinate TMM-F, or only normalized TMM-N.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to runs/latest-<variant>.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Split, training and sampling seed (default: 42).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--references", type=int, default=100)
    parser.add_argument("--samples", type=int, default=300, help="TMM draws per test trajectory.")
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--val-cases", type=int, default=50)
    parser.add_argument("--max-trajectories", type=int, help="Limit independent trajectories before splitting.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(8, get_num_threads()), help="Frechet worker threads.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "runs" / f"latest-{args.variant}"
    if not args.data.is_file():
        parser.error(f"dataset not found: {args.data}; run generate_trajectories.py first")
    if min(args.epochs, args.batch_size, args.references, args.val_cases) < 1:
        parser.error("epochs, batch-size, references and val-cases must be positive")
    if min(args.samples, args.val_samples) < NUM_MIXTURES:
        parser.error(f"samples and val-samples must be at least {NUM_MIXTURES}")
    if args.max_trajectories is not None and args.max_trajectories < 10:
        parser.error("max-trajectories must be at least 10")
    if args.seed < 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("seed must be nonnegative and learning-rate must be positive and finite")
    if not 1 <= args.threads <= get_num_threads():
        parser.error(f"threads must be between 1 and {get_num_threads()}")
    cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not cuda:
        parser.error("CUDA is unavailable in this PyTorch environment; try --device cpu")
    device = torch.device("cuda" if cuda else "cpu")
    set_num_threads(args.threads)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args, device)


if __name__ == "__main__":
    main()
