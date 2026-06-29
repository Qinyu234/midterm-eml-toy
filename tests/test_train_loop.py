"""Training loop stop criteria."""
from experiment.train_loop import TrainStopConfig, training_should_stop


def test_stops_on_plateau():
    cfg = TrainStopConfig(max_steps=100, patience=5, min_delta=1e-6, osc_window=5)
    curve = [1.0, 0.5, 0.2, 0.19, 0.191, 0.1905, 0.1902, 0.1904, 0.1901, 0.1903]
    assert training_should_stop(curve, cfg)
