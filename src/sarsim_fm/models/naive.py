"""Seasonal naive baseline: repeat the last season. All quantiles equal the point forecast."""

from __future__ import annotations

import numpy as np

from sarsim_fm.runtime import Runtime


class SeasonalNaive:
    def __init__(self, season_length: int | None):
        # None means "use each eval set's own season_length".
        self.season_length = season_length

    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        out = np.empty((len(contexts), horizon, len(levels)), dtype=np.float32)
        for i, history in enumerate(contexts):
            season = self.season_length or int(season_lengths[i])
            season = min(max(season, 1), len(history))
            point = np.tile(history[-season:], int(np.ceil(horizon / season)))[:horizon]
            out[i] = point[:, None]
        return out


def load(path: str | None, runtime: Runtime, context_length: int) -> SeasonalNaive:
    return SeasonalNaive(int(path) if path else None)
