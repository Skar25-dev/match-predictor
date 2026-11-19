import pytest

from matchpredictor.evaluation import complete_seasons, run_backtest, score_table
from matchpredictor.evaluation.backtest import main
from matchpredictor.models import EloLogit, FrequencyBaseline
from tests.synthetic import synthetic_games, write_as_csvs


@pytest.fixture(scope="module")
def games():
    return synthetic_games()


class _Spy:
    """Modelo que registra con qué temporadas se ha entrenado."""

    trained_on: list[set] = []

    def fit(self, games):
        _Spy.trained_on.append(set(games["season"]))
        self.inner = FrequencyBaseline().fit(games)
        return self

    def predict_proba(self, games):
        return self.inner.predict_proba(games)


def test_each_season_is_predicted_only_with_earlier_seasons(games):
    _Spy.trained_on = []

    run_backtest(games, {"spy": _Spy}, ["2022-23", "2023-24"], include_market=False)

    assert _Spy.trained_on == [
        {"2019-20", "2020-21", "2021-22"},
        {"2019-20", "2020-21", "2021-22", "2022-23"},
    ]


def test_season_without_history_is_skipped_and_no_seasons_is_an_error(games):
    predictions = run_backtest(games, {"f": FrequencyBaseline}, ["2019-20", "2020-21"])
    assert set(predictions["season"]) == {"2020-21"}

    with pytest.raises(ValueError, match="historia previa"):
        run_backtest(games, {"f": FrequencyBaseline}, ["2019-20"])


def test_predictions_have_one_row_per_match_and_model(games):
    predictions = run_backtest(
        games, {"f": FrequencyBaseline, "e": EloLogit}, ["2023-24"], include_market=True
    )

    season_matches = (games["season"] == "2023-24").sum()
    assert predictions.groupby("model").size().to_dict() == {
        "e": season_matches,
        "f": season_matches,
        "market": season_matches,
    }
    assert predictions[["p_home", "p_draw", "p_away"]].sum(axis=1).tolist() == pytest.approx(
        [1.0] * len(predictions)
    )


def test_score_table_compares_only_matches_all_models_share(games):
    no_odds = games.copy()
    in_test_season = no_odds.index[no_odds["season"] == "2023-24"]
    no_odds.loc[in_test_season[:100], ["b365_home", "b365_draw", "b365_away"]] = float("nan")
    test_season = no_odds[no_odds["season"] == "2023-24"]
    expected_common = test_season["b365_home"].notna().sum()

    predictions = run_backtest(no_odds, {"f": FrequencyBaseline}, ["2023-24"])
    table = score_table(predictions)

    assert 0 < expected_common < len(test_season)  # el hueco de cuotas existe de verdad
    assert set(table["matches"]) == {expected_common}
    assert set(table["model"]) == {"f", "market"}
    all_matches = score_table(predictions, common=False).set_index("model")["matches"]
    assert all_matches["f"] == len(test_season)


def test_score_table_groups_by_season(games):
    predictions = run_backtest(games, {"f": FrequencyBaseline}, ["2022-23", "2023-24"])

    table = score_table(predictions, by=["season"])

    assert table[["season", "model"]].values.tolist() == [
        ["2022-23", "f"],
        ["2022-23", "market"],
        ["2023-24", "f"],
        ["2023-24", "market"],
    ]


def test_market_is_a_strong_benchmark_and_elo_beats_frequencies(games):
    predictions = run_backtest(
        games, {"frecuencias": FrequencyBaseline, "elo": EloLogit}, ["2022-23", "2023-24"]
    )

    table = score_table(predictions).set_index("model")["log_loss"]

    assert table["market"] < table["elo"] < table["frecuencias"]


def test_complete_seasons_excludes_partial_ones(games):
    partial = games[~((games["season"] == "2023-24") & (games.index % 3 != 0))]

    assert complete_seasons(partial) == ["2019-20", "2020-21", "2021-22", "2022-23"]


def test_cli_runs_end_to_end(tmp_path, capsys, games):
    write_as_csvs(games, tmp_path)

    assert main(["--data", str(tmp_path), "--last", "2"]) == 0

    out = capsys.readouterr().out
    assert "2023-24" in out and "2022-23" in out
    for model in ("frecuencias", "elo_logit", "market"):
        assert model in out
    assert "GLOBAL" in out and "POR DIVISIÓN" in out
