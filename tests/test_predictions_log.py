import math

import pandas as pd
import pytest

from matchpredictor.evaluation import predictions_log as plog
from matchpredictor.service import PredictorService
from tests.synthetic import add_shots, synthetic_fixtures, synthetic_raw_two_divisions


@pytest.fixture(scope="module")
def raw():
    return add_shots(synthetic_raw_two_divisions())


def _live_log(raw, made_on="2030-01-01"):
    entries = PredictorService(raw).predict_fixtures(synthetic_fixtures(raw))
    return plog.entries_to_log(entries, made_on)


def test_entries_to_log_skips_errors_and_keeps_probabilities(raw):
    log = _live_log(raw)

    assert len(log) == 3  # el cuarto partido tiene un equipo desconocido
    assert list(log.columns) == plog.LOG_COLUMNS
    assert (log["mode"] == "live").all() and (log["made_on"] == "2030-01-01").all()
    assert log[plog.PROB_COLUMNS].sum(axis=1).tolist() == pytest.approx([1.0] * 3, abs=2e-4)
    no_odds = log[log["home_team"] == "U01"].iloc[0]
    assert pd.isna(no_odds["m_home"]) and pd.isna(no_odds["market_source"])


def test_append_keeps_the_first_prediction_of_each_match(tmp_path, raw):
    path = tmp_path / "log.csv"
    first = _live_log(raw, "2030-01-01")

    assert plog.append_to_log(path, first) == 3
    later = first.assign(made_on="2030-01-05", p_home=0.99, p_draw=0.005, p_away=0.005)
    assert plog.append_to_log(path, later) == 0  # ya registrados: no se tocan

    stored = plog.read_log(path)
    assert len(stored) == 3
    assert (stored["made_on"] == "2030-01-01").all() and (stored["p_home"] < 0.9).all()


def test_append_adds_only_new_matches(tmp_path, raw):
    path = tmp_path / "log.csv"
    first = _live_log(raw)
    plog.append_to_log(path, first.iloc[:2])

    assert plog.append_to_log(path, first) == 1
    assert len(plog.read_log(path)) == 3


def test_read_missing_log_is_empty(tmp_path):
    log = plog.read_log(tmp_path / "no_existe.csv")

    assert log.empty and list(log.columns) == plog.LOG_COLUMNS


def test_settle_matches_only_played_games(raw):
    played = raw.iloc[[5, 6]]
    log = pd.DataFrame(
        {
            "made_on": "2030-01-01",
            "mode": "live",
            "division": played["division"].to_numpy(),
            "date": played["date"].dt.date.astype(str).to_numpy(),
            "home_team": played["home_team"].to_numpy(),
            "away_team": played["away_team"].to_numpy(),
            "model": "elo",
            "p_home": 0.5,
            "p_draw": 0.3,
            "p_away": 0.2,
            "market_source": "b365",
            "m_home": 0.4,
            "m_draw": 0.3,
            "m_away": 0.3,
        }
    )
    pending = _live_log(raw)

    settled = plog.settle(pd.concat([log, pending], ignore_index=True), raw)

    assert len(settled) == 2
    assert settled["result"].tolist() == played["result"].tolist()


def test_score_compares_model_and_market_on_the_same_matches():
    settled = pd.DataFrame(
        {
            "mode": ["live", "live", "live"],
            "result": ["H", "D", "A"],
            "p_home": [0.8, 0.2, 0.2],
            "p_draw": [0.1, 0.6, 0.2],
            "p_away": [0.1, 0.2, 0.6],
            "m_home": [0.4, 0.3, 0.3],
            "m_draw": [0.3, 0.4, 0.3],
            "m_away": [0.3, 0.3, 0.4],
        }
    )
    extra = settled.iloc[[0]].assign(m_home=None, m_draw=None, m_away=None)

    table = plog.score(pd.concat([settled, extra], ignore_index=True))
    todos = table[table["registro"] == "todos"].set_index("predictor")

    assert todos.loc["modelo", "partidos"] == 3  # sin cuotas no entra
    expected_model = -sum(pd.Series([0.8, 0.6, 0.6]).map(math.log)) / 3
    assert todos.loc["modelo", "log_loss"] == pytest.approx(expected_model)
    expected_market = -sum(pd.Series([0.4, 0.4, 0.4]).map(math.log)) / 3
    assert todos.loc["mercado", "log_loss"] == pytest.approx(expected_market)
    assert todos.loc["modelo", "log_loss"] < todos.loc["mercado", "log_loss"]


def test_replay_never_uses_the_future_and_uses_opening_odds(raw):
    since = raw["date"].sort_values().iloc[len(raw) * 4 // 5]
    with_closing = raw.assign(avgc_home=1.5, avgc_draw=4.0, avgc_away=6.0)

    log = plog.replay(with_closing, since.date().isoformat(), step_days=30)

    assert not log.empty and (log["mode"] == "replay").all()
    assert (log["date"] >= since.date().isoformat()).all()
    assert since.date().isoformat() in set(log["date"])  # el primer día cuenta
    assert (log["made_on"] <= log["date"]).all()
    assert set(log["market_source"].dropna()) == {"b365"}  # cierre descartado
    assert log[plog.PROB_COLUMNS].sum(axis=1).tolist() == pytest.approx([1.0] * len(log), abs=2e-4)


def test_replay_prediction_equals_a_model_trained_only_on_the_past(raw):
    since = raw["date"].sort_values().iloc[len(raw) * 4 // 5]
    log = plog.replay(raw, since.date().isoformat(), step_days=10_000)  # un solo bloque

    row = log.iloc[0]
    past = raw[raw["date"] < since]
    direct = PredictorService(past).predict(row["home_team"], row["away_team"], row["division"])

    assert row["p_home"] == pytest.approx(direct["probabilities"]["home"], abs=1e-4)
    assert (log["made_on"] == since.date().isoformat()).all()


def test_replay_scores_end_to_end(raw):
    since = raw["date"].sort_values().iloc[len(raw) * 4 // 5]

    log = plog.replay(raw, since.date().isoformat())
    table = plog.score(plog.settle(log, raw))

    assert set(table["predictor"]) == {"modelo", "mercado"}
    assert (table["log_loss"] > 0.5).all() and (table["log_loss"] < 1.3).all()


def test_cli_record_replay_and_report(tmp_path, capsys, raw):
    from tests.synthetic import write_as_csvs

    data = tmp_path / "raw"
    data.mkdir()
    write_as_csvs(raw, data)
    log = tmp_path / "log.csv"
    since = raw["date"].sort_values().iloc[len(raw) * 4 // 5].date().isoformat()
    base = ["--data", str(data), "--log", str(log)]

    assert plog.main(["replay", *base, "--since", since]) == 0
    assert plog.main(["report", *base]) == 0
    out = capsys.readouterr().out
    assert "modelo" in out and "mercado" in out and "Partidos con resultado" in out

    assert plog.main(["replay", *base]) == 1  # falta --since
    assert plog.main(["record", *base, "--fixtures", str(tmp_path / "no.csv")]) == 1


def test_report_without_settled_matches(tmp_path, capsys, raw):
    from tests.synthetic import write_as_csvs

    data = tmp_path / "raw"
    data.mkdir()
    write_as_csvs(raw, data)

    assert plog.main(["report", "--data", str(data), "--log", str(tmp_path / "x.csv")]) == 0
    assert "Aún no hay" in capsys.readouterr().out
