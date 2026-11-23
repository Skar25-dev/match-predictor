import warnings

import numpy as np
import pandas as pd
import pytest

from matchpredictor.evaluation import log_loss
from matchpredictor.models import FeatureLogit, FrequencyBaseline, GradientBoosting

PROBS = ["p_home", "p_draw", "p_away"]


def signal_data(n=4000, seed=0):
    """``x1`` determina el resultado (más alto = más victorias locales); ``x2`` es ruido."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    home = 0.74 / (1 + np.exp(-1.8 * x1))
    probs = np.column_stack([home, np.full(n, 0.26), 0.74 - home])
    result = [rng.choice(["H", "D", "A"], p=row) for row in probs]
    return pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, n), "division": "SP1", "result": result})


@pytest.fixture(scope="module")
def data():
    return signal_data()


@pytest.mark.parametrize(
    "factory",
    [lambda: FeatureLogit(["x1", "x2"]), lambda: GradientBoosting(["x1", "x2"])],
    ids=["logit", "gbm"],
)
class TestBothModels:
    def test_probabilities_are_valid(self, factory, data):
        probs = factory().fit(data).predict_proba(data)

        assert list(probs.columns) == PROBS
        assert probs.sum(axis=1).tolist() == pytest.approx([1.0] * len(probs))
        assert probs.index.equals(data.index)

    def test_strong_feature_raises_home_win_probability(self, factory, data):
        model = factory().fit(data)
        low = data.head(1).assign(x1=-2.0, x2=0.0)
        high = data.head(1).assign(x1=2.0, x2=0.0)

        p_low, p_high = model.predict_proba(low).iloc[0], model.predict_proba(high).iloc[0]

        assert p_high["p_home"] > 0.5 > p_low["p_home"]
        assert p_high["p_away"] < p_low["p_away"]  # el orden de las columnas es el correcto

    def test_beats_frequencies_out_of_sample(self, factory, data):
        train, test = data.iloc[:3000], data.iloc[3000:]

        model = log_loss(factory().fit(train).predict_proba(test), test["result"])
        baseline = log_loss(FrequencyBaseline().fit(train).predict_proba(test), test["result"])

        assert model < baseline - 0.05

    def test_missing_feature_values_are_accepted(self, factory, data):
        holey = data.copy()
        holey.loc[holey.index[::5], "x1"] = np.nan

        probs = factory().fit(holey).predict_proba(holey)

        assert not probs.isna().any().any()

    def test_entirely_empty_feature_column_is_accepted_without_warnings(self, factory, data):
        empty = data.assign(x2=np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            probs = factory().fit(empty).predict_proba(empty)

        assert not probs.isna().any().any()

    def test_unknown_feature_is_a_clear_error(self, factory, data):
        model = factory()
        model.features = ["x1", "nope"]

        with pytest.raises(ValueError, match="nope"):
            model.fit(data)


def test_logit_replaces_missing_values_with_the_training_mean(data):
    model = FeatureLogit(["x1", "x2"]).fit(data)
    row = data.head(1).assign(x2=0.0)

    missing = model.predict_proba(row.assign(x1=np.nan))
    at_mean = model.predict_proba(row.assign(x1=model.mean_[0]))

    pd.testing.assert_frame_equal(missing, at_mean)


def test_logit_division_matters():
    frame = signal_data(1000).assign(division=lambda d: np.where(d.index % 2 == 0, "SP1", "SP2"))
    model = FeatureLogit(["x1"]).fit(frame)
    row = frame.head(1)

    assert not model.predict_proba(row.assign(division="SP1")).equals(
        model.predict_proba(row.assign(division="SP2"))
    )
