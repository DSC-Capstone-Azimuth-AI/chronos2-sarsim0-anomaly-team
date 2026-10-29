from collections import Counter

import numpy as np
import pytest

from sarsim_fm.config import load_config
from sarsim_fm.data.anomaly_injection import KINDS, clean_rejection_reason, inject_anomaly


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_nonmutating_intervals(kind):
    x = np.sin(np.arange(200) / 3) + 5
    original = x.copy()
    a, label = inject_anomaly(x, np.random.default_rng(7), kind, 5, 24, 100)
    b, other = inject_anomaly(x, np.random.default_rng(7), kind, 5, 24, 100)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(a[:100], x[:100])
    np.testing.assert_array_equal(a[124:], x[124:])
    assert label == other and (label.start, label.end) == (100, 124)
    assert np.isfinite(a).all()


def test_filters_and_seed_ranges():
    assert clean_rejection_reason([np.nan], 1) == "non_finite"
    assert clean_rejection_reason([-1e6, 1], 1) == "clipped"
    assert clean_rejection_reason([0, 0, 1], 2) == "all_zero_context"
    dev = load_config("configs/data/sim-anomaly-dev.yaml")
    test = load_config("configs/data/sim-anomaly-test.yaml")
    seeds = [set(range(s["seed"], s["seed"] + 30))
             for s in [*dev["splits"].values(), *test["splits"].values()]]
    assert all(not a & b for i, a in enumerate(seeds) for b in seeds[i + 1:])


def test_retries_recorded_and_bounded():
    from sarsim_fm.data.sim import draw_clean_series

    class Fake:
        calls = 0

        def generate_batch(self, batch_size):
            self.calls += 1
            return np.array([[0, 0] if self.calls == 1 else [1, 2]]), np.array([[3, 4]])

    counts = Counter()
    assert len(draw_clean_series(Fake(), 4, 2, 3, counts)) == 4
    assert counts == {"all_zero_context": 1}
    with pytest.raises(ValueError, match="retries"):
        draw_clean_series(Fake(), 4, 2, 1, Counter())


def test_generation_manifests_are_deterministic(tmp_path, monkeypatch):
    import json

    from sarsim_fm.data import sim

    class Fake:
        def __init__(self, config, seed):
            self.config, self.rng = config, np.random.default_rng(seed)

        def generate_batch(self, batch_size):
            return (self.rng.normal(size=(1, self.config.context_window)),
                    self.rng.normal(size=(1, self.config.prediction_window)))

    monkeypatch.setattr(sim, "SarSim0Generator", Fake)
    spec = {"n_series": 2, "series_length": 100, "clean_prefix_length": 50,
            "context_length": 20, "season_length": 12, "seed": 3100000,
            "severity_range": [3, 4], "duration_periods": [0.2, 1.0]}
    a = sim.make_anomaly_split("sim-anomaly-dev", spec, tmp_path / "a")
    b = sim.make_anomaly_split("sim-anomaly-dev", spec, tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()
    assert a.with_suffix(".manifest.json").read_bytes() == b.with_suffix(".manifest.json").read_bytes()
    reasons = json.loads(a.with_suffix(".manifest.json").read_text())["rejections"]
    assert set(reasons) == {"non_finite", "clipped", "all_zero_context", "too_short"}
    out, label = inject_anomaly(np.arange(30), np.random.default_rng(1), "spike", 3, 1, 0)
    assert label.start == 0 and label.end == 1 and np.isfinite(out).all()
