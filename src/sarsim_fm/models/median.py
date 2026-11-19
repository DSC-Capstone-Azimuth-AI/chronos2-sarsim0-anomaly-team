"""Rolling median point forecast; uncertainty is calibrated through residual scores."""
import numpy as np


class RollingMedian:
    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        return np.broadcast_to(np.array([np.median(x) for x in contexts])[:, None, None],
                               (len(contexts), horizon, len(levels))).copy()


def load(path, runtime, context_length):
    if path:
        raise ValueError("rolling_median accepts no checkpoint or period")
    return RollingMedian()
