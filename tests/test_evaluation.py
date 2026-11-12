
import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.evaluation import evaluate_model


def test_perfect_predictions():
    actual = [100, 200, 300]
    predicted = [100, 200, 300]

    result = evaluate_model(actual, predicted)

    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)
    assert result["R2"] == pytest.approx(1.0)


def test_evaluation_metrics():
    actual = np.array([100, 200, 300, 400])
    predicted = np.array([110, 190, 320, 380])

    result = evaluate_model(actual, predicted)

    assert result["MAE"] == pytest.approx(
        mean_absolute_error(actual, predicted)
    )
    assert result["RMSE"] == pytest.approx(
        np.sqrt(mean_squared_error(actual, predicted))
    )
    assert result["R2"] == pytest.approx(
        r2_score(actual, predicted)
    )


def test_all_metrics_are_finite():
    result = evaluate_model(
        [80, 120, 200],
        [90, 130, 180],
    )

    assert all(np.isfinite(value) for value in result.values())
