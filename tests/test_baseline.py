import pandas as pd
import pytest

from matchpredictor.evaluation import log_loss
from matchpredictor.models import EloLogit, FrequencyBaseline
from tests.synthetic import synthetic_games

PROBS = ["p_home", "p_draw", "p_away"]


@pytest.fixture(scope="module")
def games():
    return synthetic_games()


def test_frequency_baseline_predicts_training_frequencies(games):
    model = FrequencyBaseline().fit(games)

    probs = model.predict_proba(games.head(4))

    shares = games["result"].value_counts(normalize=True)
    assert probs.iloc[0].tolist() == pytest.approx([shares["H"], shares["D"], shares["A"]])
    assert probs.index.tolist() == games.head(4).index.tolist()


def test_elo_logit_probabilities_are_valid(games):
    probs = EloLogit().fit(games).predict_proba(games)

    assert list(probs.columns) == PROBS
    assert probs.sum(axis=1).tolist() == pytest.approx([1.0] * len(probs))
    assert ((probs > 0) & (probs < 1)).all().all()


def test_elo_logit_favours_the_stronger_team(games):
    model = EloLogit().fit(games)
    weak_home = games.head(1).assign(elo_diff=-300.0)
    strong_home = games.head(1).assign(elo_diff=300.0)

    weak, strong = model.predict_proba(weak_home), model.predict_proba(strong_home)

    assert strong["p_home"].iloc[0] > weak["p_home"].iloc[0]
    assert strong["p_away"].iloc[0] < weak["p_away"].iloc[0]


def test_elo_logit_beats_frequencies_out_of_sample(games):
    train = games[games["season"] < "2023-24"]
    test = games[games["season"] >= "2023-24"]

    elo = log_loss(EloLogit().fit(train).predict_proba(test), test["result"])
    frequency = log_loss(FrequencyBaseline().fit(train).predict_proba(test), test["result"])

    assert elo < frequency


def test_division_changes_the_prediction(games):
    model = EloLogit().fit(
        games.assign(division=["SP1", "SP2"] * (len(games) // 2) + ["SP1"] * (len(games) % 2))
    )
    row = games.head(1)

    sp1 = model.predict_proba(row.assign(division="SP1"))
    sp2 = model.predict_proba(row.assign(division="SP2"))

    assert not sp1.equals(sp2)
    assert isinstance(sp1, pd.DataFrame)
