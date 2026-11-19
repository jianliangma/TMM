"""MLP-Mixer mixture density network and kernel trajectory decoding."""

import math

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.nn import functional as F


INDUCING_STEPS = np.arange(0, 26, 5)
NUM_COEFFICIENTS = 3 * len(INDUCING_STEPS)
NUM_MIXTURES = 5


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, dim=64, token_dim=128, channel_dim=128):
        super().__init__()
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mixing = nn.Sequential(nn.Linear(dim, channel_dim), nn.GELU(), nn.Linear(channel_dim, dim))
        self.token_norm = nn.LayerNorm(dim)
        self.token_mixing = nn.Sequential(nn.Linear(num_tokens, token_dim), nn.GELU(), nn.Linear(token_dim, num_tokens))

    def forward(self, tokens):
        tokens = tokens + self.channel_mixing(self.channel_norm(tokens))
        return tokens + self.token_mixing(self.token_norm(tokens).transpose(1, 2)).transpose(1, 2)


class MDN(nn.Module):
    def __init__(self, input_size, dim=64):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1, dim), nn.ReLU())
        self.mixer_layers = nn.Sequential(MixerBlock(input_size, dim), MixerBlock(input_size, dim))
        self.norm = nn.LayerNorm(dim)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(input_size * dim, NUM_COEFFICIENTS * NUM_MIXTURES)
        self.fc_sigma = nn.Linear(input_size * dim, NUM_COEFFICIENTS * NUM_MIXTURES)
        self.fc_alpha = nn.Linear(input_size * dim, NUM_MIXTURES)

    def forward(self, features):
        tokens = self.embedding(features.unsqueeze(-1))
        hidden = self.flatten(self.norm(self.mixer_layers(tokens)))
        output_shape = (-1, NUM_COEFFICIENTS, NUM_MIXTURES)
        mean = self.fc_mu(hidden).reshape(output_shape)
        sigma = (F.elu(self.fc_sigma(hidden)) + 1.0 + 1e-6).clamp(1e-3, 10.0).reshape(output_shape)
        alpha = F.softmax(self.fc_alpha(hidden), dim=1)
        return torch.cat((mean, sigma, alpha.unsqueeze(1)), dim=1)


def mdn_loss(targets, parameters):
    mean = parameters[:, :NUM_COEFFICIENTS]
    sigma = parameters[:, NUM_COEFFICIENTS:2 * NUM_COEFFICIENTS]
    alpha = parameters[:, -1]
    standardized = (targets.unsqueeze(-1) - mean) / sigma
    log_probability = ((alpha + 1e-8).log() - 0.5 * NUM_COEFFICIENTS * math.log(2 * math.pi)
                       - (sigma.log() + 0.5 * standardized.square()).sum(dim=1))
    return -torch.logsumexp(log_probability, dim=1).mean()


def rbf_basis(steps):
    return np.exp(-0.05 * (np.asarray(steps)[:, None] - INDUCING_STEPS) ** 2)


def fit_coefficients(observations, futures):
    future_basis = rbf_basis(np.arange(1, futures.shape[1] + 1))
    anchor_basis = rbf_basis([0])
    normal_matrix = (future_basis.T @ future_basis + 9e5 * (anchor_basis.T @ anchor_basis)
                     + 0.1 * np.eye(len(INDUCING_STEPS)))
    projection = np.linalg.solve(normal_matrix, future_basis.T)
    coefficients = projection @ (futures - observations[:, -1:])
    return coefficients.swapaxes(1, 2).reshape(len(futures), NUM_COEFFICIENTS).astype(np.float32)


def sample_modes(parameters, basis, anchor, rng, sample_count, rotation=None, scale=1.0):
    probabilities = parameters[-1].astype(np.float64)
    components = rng.choice(NUM_MIXTURES, p=probabilities / probabilities.sum(), size=sample_count)
    means = parameters[:NUM_COEFFICIENTS].T[components]
    sigmas = parameters[NUM_COEFFICIENTS:2 * NUM_COEFFICIENTS].T[components]
    coefficients = rng.normal(means, sigmas).reshape(sample_count, 3, len(INDUCING_STEPS))
    offsets = basis @ coefficients.swapaxes(1, 2)
    if rotation is not None:
        offsets = (offsets @ rotation) / scale
    trajectories = offsets + anchor
    assignments = GaussianMixture(n_components=NUM_MIXTURES, random_state=0).fit_predict(trajectories[:, -1])
    modes = [trajectories[assignments == label] for label in np.unique(assignments)]
    modes.sort(key=len, reverse=True)
    return np.stack([mode.mean(axis=0) for mode in modes]), np.array([len(mode) / sample_count for mode in modes])
