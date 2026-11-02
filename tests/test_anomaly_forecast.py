import numpy as np

from sarsim_fm.anomaly.forecast import block_forecast, cached_forecast, rolling_forecast
from sarsim_fm.evaluate import LEVELS


class CausalFake:
    season_length = 1

    def __init__(self):
        self.origins = []

    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        self.origins.extend(int(x[-1]) + 1 for x in contexts)
        assert len(contexts) <= batch_size
        for x in contexts:
            np.testing.assert_array_equal(x, np.arange(x[-1] - len(x) + 1, x[-1] + 1))
        return np.array([np.full((horizon, len(levels)), x[-1]) for x in contexts])


def test_rolling_and_block_alignment():
    x, fake = np.arange(13), CausalFake()
    r = rolling_forecast(fake, "a", x, 4, 3, LEVELS, 2)
    assert fake.origins == list(range(4, 13))
    np.testing.assert_array_equal(r.actual, x[r.time])
    np.testing.assert_array_equal(r.quantiles[:, 0], x[r.time - 1])
    assert (r.forecast_horizon == 1).all()
    fake = CausalFake()
    b = block_forecast(fake, "a", x, 4, 3, 4, LEVELS, 2)
    assert fake.origins == [4, 8, 12]
    np.testing.assert_array_equal(b.forecast_horizon, [1, 2, 3, 4, 1, 2, 3, 4, 1])
    np.testing.assert_array_equal(b.actual, x[b.time])


def test_cache_reuse_and_stale_rejection(tmp_path):
    path, fake = tmp_path / "a.parquet", CausalFake()
    kw = {"model": "fake:sha", "start": 4, "context_length": 3, "levels": LEVELS, "batch_size": 2}
    cached_forecast(path, fake, "a", np.arange(13), **kw)
    fake.origins.clear()
    cached_forecast(path, fake, "a", np.arange(13), **kw)
    assert not fake.origins
    cached_forecast(path, fake, "a", np.arange(14), **kw)
    assert fake.origins
    fake.origins.clear()
    cached_forecast(path, fake, "a", np.arange(14), **{**kw, "model": "fake:new-sha"})
    assert fake.origins
