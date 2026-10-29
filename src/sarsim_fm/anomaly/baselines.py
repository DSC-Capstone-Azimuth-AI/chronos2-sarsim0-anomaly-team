"""Simple detector forecasters using exclusively each supplied past context."""

from statistics import NormalDist

import numpy as np


class MedianMAD:
    """Rolling median with Gaussian quantiles from local MAD; no fitting on test data."""

    def __init__(self, season_length=1):
        self.season_length = season_length

    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        z = np.array([NormalDist().inv_cdf(p) for p in levels])
        if len({len(x) for x in contexts}) == 1:
            matrix = np.asarray(contexts)
            median = np.median(matrix, axis=1)
            scale = np.maximum(1.4826 * np.median(np.abs(matrix - median[:, None]), axis=1), 1e-8)
            q = median[:, None] + z * scale[:, None]
            return np.repeat(q[:, None, :], horizon, axis=1)
        result = []
        for x in contexts:
            median = np.median(x)
            scale = max(1.4826 * np.median(np.abs(x - median)), 1e-8)
            result.append(np.tile(median + z * scale, (horizon, 1)))
        return np.asarray(result)
