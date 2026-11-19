"""Trajectory normalization, discrete Frechet features and kinematic baselines."""

import numpy as np
from numba import njit, prange


def normalize_trajectories(observations, futures):
    origins = observations[:, :1]
    centered = observations - origins
    rotations = np.repeat(np.eye(3)[None], len(observations), axis=0)
    axis_x = np.array([1.0, 0.0, 0.0])

    for index, track in enumerate(centered):
        moving = np.flatnonzero(np.linalg.norm(track, axis=1) > 1e-8)
        if not len(moving):
            continue
        direction = track[moving[0]] / np.linalg.norm(track[moving[0]])
        axis = np.cross(direction, axis_x)
        sine = np.linalg.norm(axis)
        cosine = direction[0]
        if sine < 1e-8:
            if cosine < 0:
                rotations[index] = np.diag([-1.0, -1.0, 1.0])
            continue
        axis /= sine
        axis_x_component, axis_y_component, axis_z_component = axis
        skew = np.array([
            [0, -axis_z_component, axis_y_component],
            [axis_z_component, 0, -axis_x_component],
            [-axis_y_component, axis_x_component, 0],
        ])
        rotations[index] += sine * skew + (1 - cosine) * (skew @ skew)

    lengths = np.linalg.norm(np.diff(centered, axis=1), axis=2).sum(axis=1)
    scales = np.ones(len(lengths))
    np.divide(10.0, lengths, out=scales, where=lengths > 1e-6)
    transpose = rotations.swapaxes(1, 2)
    normalized_obs = (centered @ transpose) * scales[:, None, None]
    normalized_future = ((futures - origins) @ transpose) * scales[:, None, None]
    return normalized_obs, normalized_future, rotations, scales


@njit(cache=True)
def _frechet_distance(left, right, buffer):
    for row in range(len(left)):
        diagonal = np.inf
        for col in range(len(right)):
            up = buffer[col] if row else np.inf
            previous = buffer[col - 1] if col else np.inf
            delta_x = left[row, 0] - right[col, 0]
            delta_y = left[row, 1] - right[col, 1]
            delta_z = left[row, 2] - right[col, 2]
            squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
            buffer[col] = squared if row == 0 and col == 0 else max(squared, min(up, previous, diagonal))
            diagonal = up
    return np.sqrt(buffer[len(right) - 1])


@njit(parallel=True, cache=True)
def training_distances(observations):
    size = len(observations)
    distances = np.zeros((size, size), dtype=np.float64)
    for row in prange(size):
        buffer = np.empty(observations.shape[1])
        for col in range(row + 1, size):
            distance = _frechet_distance(observations[row], observations[col], buffer)
            distances[row, col] = distance
            distances[col, row] = distance
    return distances


@njit(parallel=True, cache=True)
def cross_distances(observations, references):
    distances = np.empty((len(observations), len(references)))
    for row in prange(len(observations)):
        buffer = np.empty(references.shape[1])
        for col in range(len(references)):
            distances[row, col] = _frechet_distance(observations[row], references[col], buffer)
    return distances


def nearest_neighbor_references(distances, count):
    size = len(distances)
    order = np.empty(size, dtype=int)
    visited = np.zeros(size, dtype=bool)
    index = 0
    for step in range(size):
        order[step] = index
        visited[index] = True
        if step + 1 < size:
            index = np.argmin(np.where(visited, np.inf, distances[index]))
    return order[np.linspace(0, size - 1, min(count, size), dtype=int)]


def kinematic_forecasts(observation, horizon):
    steps = np.arange(1, horizon + 1)[:, None]
    last = observation[-1]
    velocity = last - observation[-2]
    acceleration = velocity - (observation[-2] - observation[-3])
    cv = last + velocity * steps
    ca = cv + 0.5 * acceleration * steps ** 2
    return cv, ca


def trajectory_errors(prediction, truth):
    point_errors = np.linalg.norm(prediction - truth, axis=1)
    distance = _frechet_distance(prediction, truth, np.empty(len(truth)))
    return np.array([point_errors.mean(), point_errors[-1], distance])
