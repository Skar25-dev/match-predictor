import math

import numpy as np
import pandas as pd
import pytest

from matchpredictor.evaluation import (
    brier_score,
    implied_probabilities,
    log_loss,
    market_probabilities,
)


def _probs(rows):
    return pd.DataFrame(rows, columns=["p_home", "p_draw", "p_away"])


UNIFORM = _probs([[1 / 3, 1 / 3, 1 / 3]] * 3)
RESULTS = pd.Series(["H", "D", "A"])


def test_uniform_predictions_give_reference_values():
    assert log_loss(UNIFORM, RESULTS) == pytest.approx(math.log(3))
    assert brier_score(UNIFORM, RESULTS) == pytest.approx(2 / 3)


def test_perfect_predictions_score_zero():
    perfect = _probs([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert brier_score(perfect, RESULTS) == 0.0
    assert log_loss(perfect, RESULTS) == pytest.approx(0.0, abs=1e-12)


def test_confident_wrong_prediction_is_penalised_but_finite():
    wrong = _probs([[0, 0, 1]])

    assert log_loss(wrong, pd.Series(["H"])) == pytest.approx(-math.log(1e-15))
    assert brier_score(wrong, pd.Series(["H"])) == 2.0


def test_known_log_loss_value():
    probs = _probs([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])

    value = log_loss(probs, pd.Series(["H", "A"]))

    assert value == pytest.approx(-(math.log(0.5) + math.log(0.7)) / 2)


def test_column_order_matters():
    probs = _probs([[0.7, 0.2, 0.1]])

    assert log_loss(probs, pd.Series(["H"])) < log_loss(probs, pd.Series(["A"]))


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError, match="NaN"):
        log_loss(_probs([[np.nan, 0.5, 0.5]]), pd.Series(["H"]))
    with pytest.raises(ValueError, match="no válidos"):
        log_loss(_probs([[0.3, 0.3, 0.4]]), pd.Series(["X"]))
    with pytest.raises(ValueError, match="longitud"):
        brier_score(UNIFORM, pd.Series(["H"]))


def test_implied_probabilities_remove_margin():
    probs = implied_probabilities(pd.Series([2.0]), pd.Series([4.0]), pd.Series([4.0]))
    assert probs.iloc[0].tolist() == [0.5, 0.25, 0.25]

    with_margin = implied_probabilities(pd.Series([1.9]), pd.Series([3.5]), pd.Series([4.0]))
    assert with_margin.iloc[0].sum() == pytest.approx(1.0)
    # 1/1.9 + 1/3.5 + 1/4 = 1.0617: cada probabilidad bruta se reduce al normalizar
    assert with_margin.iloc[0]["p_home"] == pytest.approx((1 / 1.9) / (1 / 1.9 + 1 / 3.5 + 0.25))
    assert with_margin.iloc[0]["p_home"] < 1 / 1.9


def test_implied_probabilities_invalid_odds_become_nan():
    probs = implied_probabilities(
        pd.Series([2.0, np.nan, 1.0]), pd.Series([3.0, 3.0, 3.0]), pd.Series([4.0, 4.0, 4.0])
    )

    assert probs.iloc[0].notna().all()
    assert probs.iloc[1].isna().all()
    assert probs.iloc[2].isna().all()  # cuota 1.0: no es una cuota válida


def test_market_probabilities_use_first_available_source_and_label_it():
    df = pd.DataFrame(
        {
            "avgc_home": [2.0, np.nan, np.nan],
            "avgc_draw": [4.0, np.nan, np.nan],
            "avgc_away": [4.0, np.nan, np.nan],
            "b365_home": [9.0, 3.0, np.nan],
            "b365_draw": [9.0, 3.0, np.nan],
            "b365_away": [9.0, 3.0, np.nan],
        }
    )

    market = market_probabilities(df)

    assert market["market_source"].iloc[:2].tolist() == ["avgc", "b365"]
    assert pd.isna(market["market_source"].iloc[2])
    assert market.loc[0, "p_home"] == 0.5  # gana avgc sobre b365
    assert market.loc[1, "p_home"] == pytest.approx(1 / 3)
    assert market.loc[2, ["p_home", "p_draw", "p_away"]].isna().all()


def test_market_probabilities_without_any_odds_columns():
    market = market_probabilities(pd.DataFrame({"home_team": ["A"]}))

    assert market["p_home"].isna().all()
