"""Unit tests for the min-max normalizer (Problem 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcg.core.constants import METRIC_ORDER
from pcg.core.exceptions import MetricSchemaError, NormalizerNotFittedError
from pcg.data.normalizer import MinMaxNormalizer


def _frame(values_per_metric: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(values_per_metric.values())))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame(values_per_metric, index=idx)


def test_fit_transform_maps_to_unit_range() -> None:
    df = _frame({m: [0.0, 50.0, 100.0] for m in METRIC_ORDER})
    out = MinMaxNormalizer().fit_transform(df)
    for m in METRIC_ORDER:
        assert out[m].iloc[0] == pytest.approx(0.0)
        assert out[m].iloc[1] == pytest.approx(0.5)
        assert out[m].iloc[2] == pytest.approx(1.0)


def test_transform_uses_training_range_not_inference_range() -> None:
    """Critical MLOps invariant: inference must reuse the *fitted* range
    so that distributional shift between train and serve is observable."""
    train = _frame({m: [0.0, 100.0] for m in METRIC_ORDER})
    norm = MinMaxNormalizer().fit(train)

    serve = _frame({m: [50.0, 150.0] for m in METRIC_ORDER})
    out = norm.transform(serve)

    for m in METRIC_ORDER:
        assert out[m].iloc[0] == pytest.approx(0.5)
        # Out-of-training-range value MUST stay >1 — this signal is what
        # tells the autoencoder to flag an anomaly (Problem 2).
        assert out[m].iloc[1] == pytest.approx(1.5)


def test_transform_before_fit_raises() -> None:
    df = _frame({m: [0.0, 1.0] for m in METRIC_ORDER})
    with pytest.raises(NormalizerNotFittedError):
        MinMaxNormalizer().transform(df)


def test_missing_column_raises_on_fit() -> None:
    df = _frame({m: [0.0, 1.0] for m in METRIC_ORDER[:-1]})
    with pytest.raises(MetricSchemaError):
        MinMaxNormalizer().fit(df)


def test_missing_column_raises_on_transform() -> None:
    train = _frame({m: [0.0, 1.0] for m in METRIC_ORDER})
    norm = MinMaxNormalizer().fit(train)
    bad = _frame({m: [0.0, 1.0] for m in METRIC_ORDER[:-1]})
    with pytest.raises(MetricSchemaError):
        norm.transform(bad)


def test_save_load_roundtrip(tmp_path) -> None:
    df = _frame({m: [0.0, 100.0] for m in METRIC_ORDER})
    norm = MinMaxNormalizer().fit(df)
    path = tmp_path / "norm.json"
    norm.save(path)

    restored = MinMaxNormalizer.load(path)
    for m in METRIC_ORDER:
        assert restored.mins[m] == pytest.approx(norm.mins[m])
        assert restored.maxs[m] == pytest.approx(norm.maxs[m])
    pd.testing.assert_frame_equal(norm.transform(df), restored.transform(df))


def test_constant_column_does_not_divide_by_zero() -> None:
    df = _frame({m: [0.5, 0.5, 0.5] for m in METRIC_ORDER})
    norm = MinMaxNormalizer().fit(df)
    out = norm.transform(df)
    assert np.all(np.isfinite(out.to_numpy()))


def test_load_rejects_artifact_missing_metric(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"mins": {"cpu_util": 0.0}, "maxs": {"cpu_util": 1.0}}')
    with pytest.raises(MetricSchemaError):
        MinMaxNormalizer.load(path)
