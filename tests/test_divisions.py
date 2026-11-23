import numpy as np
import pandas as pd
import pytest

from matchpredictor.evaluation.divisions import (
    bootstrap_interval,
    estimate_offset,
    first_matches,
    main,
    transitions,
)
from matchpredictor.features import compute_elo
from matchpredictor.features.elo import expected_score
from tests.synthetic import synthetic_raw_two_divisions, with_promotion, write_as_csvs


def _games(rows):
    df = pd.DataFrame(
        rows,
        columns=["season", "division", "date", "home_team", "away_team", "result"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.NA
    df["home_goals"] = df["result"].map({"H": 1, "D": 0, "A": 0})
    df["away_goals"] = df["result"].map({"H": 0, "D": 0, "A": 1})
    games, _ = compute_elo(df)
    return games


# A sube (SP2 -> SP1), B baja (SP1 -> SP2), C se queda en SP1, D en SP2, E llega nuevo a SP1
# en 2020-21 y X estuvo en SP2 en 2018-19 pero no en 2019-20 (no es un ascenso directo).
HAND = [
    ("2018-19", "SP2", "2018-09-01", "X", "D", "H"),
    ("2019-20", "SP1", "2019-09-01", "B", "C", "H"),
    ("2019-20", "SP1", "2019-09-08", "C", "B", "D"),
    ("2019-20", "SP2", "2019-09-01", "A", "D", "H"),
    ("2019-20", "SP2", "2019-09-08", "D", "A", "A"),
    ("2020-21", "SP1", "2020-09-01", "A", "C", "H"),
    ("2020-21", "SP1", "2020-09-08", "C", "A", "H"),
    ("2020-21", "SP1", "2020-09-15", "A", "E", "D"),
    ("2020-21", "SP1", "2020-09-22", "X", "C", "A"),
    ("2020-21", "SP2", "2020-09-01", "D", "B", "A"),
    ("2020-21", "SP2", "2020-09-08", "D", "B", "D"),
    ("2020-21", "SP2", "2020-09-15", "B", "D", "H"),
]


def test_transitions_find_promotions_and_relegations_only():
    found = transitions(_games(HAND))

    assert sorted(found.itertuples(index=False, name=None)) == [
        ("A", "2020-21", "promoted"),
        ("B", "2020-21", "relegated"),
    ]


def test_first_matches_respects_k_order_and_perspective():
    games = _games(HAND)

    rows = first_matches(games, k=2)

    a = rows[rows["team"] == "A"]
    assert a["actual"].tolist() == [1.0, 0.0]  # gana en casa a C; pierde en casa de C
    assert (a["sign"] == 1.0).all() and (a["kind"] == "promoted").all()
    b = rows[rows["team"] == "B"]
    assert b["actual"].tolist() == [1.0, 0.5]  # gana de visitante a D; empata con D
    assert (b["sign"] == -1.0).all()
    assert set(rows["team"]) == {"A", "B"}  # ni C, D, E ni X
    # perspectiva: el visitante usa -(elo_diff + ventaja) y el local elo_diff + ventaja
    first_home = games[(games["home_team"] == "A") & (games["season"] == "2020-21")].iloc[0]
    assert a["diff"].iloc[0] == pytest.approx(first_home["elo_diff"] + 60.0)
    away_row = games[(games["away_team"] == "B") & (games["season"] == "2020-21")].iloc[0]
    assert b["diff"].iloc[0] == pytest.approx(-(away_row["elo_diff"] + 60.0))


def test_first_matches_needs_elo_columns():
    with pytest.raises(ValueError, match="Elo"):
        first_matches(pd.DataFrame({"season": ["2020-21"]}))


def _simulate(d_true, signs, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    diff = rng.normal(0, 150, n)
    sign = signs(n)
    actual = (rng.random(n) < expected_score(diff + sign * d_true)).astype(float)
    return diff, actual, sign


def _half_and_half(n):
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


@pytest.mark.parametrize("d_true", [-80.0, 0.0, 60.0])
def test_estimate_recovers_the_true_offset(d_true):
    diff, actual, sign = _simulate(d_true, _half_and_half)

    assert estimate_offset(diff, actual, sign) == pytest.approx(d_true, abs=15)


@pytest.mark.parametrize("sign_value", [1.0, -1.0])
def test_each_kind_alone_gives_the_same_offset(sign_value):
    diff, actual, sign = _simulate(-80.0, lambda n: np.full(n, sign_value))

    assert estimate_offset(diff, actual, sign) == pytest.approx(-80.0, abs=15)


def test_estimate_returns_nan_without_data_or_without_a_solution():
    assert np.isnan(estimate_offset([], [], []))
    assert np.isnan(estimate_offset([0.0] * 50, [1.0] * 50, [1.0] * 50))  # gana siempre


def test_bootstrap_interval_contains_the_truth_and_is_reproducible():
    diff, actual, sign = _simulate(-80.0, _half_and_half, n=600, seed=3)
    rows = pd.DataFrame(
        {
            "team": np.arange(600) // 10,
            "season": "2020-21",
            "diff": diff,
            "actual": actual,
            "sign": sign,
        }
    )
    # el signo debe ser constante dentro de un equipo-temporada
    rows["sign"] = np.where(rows["team"] % 2 == 0, 1.0, -1.0)
    rows["actual"] = (
        np.random.default_rng(5).random(600) < expected_score(rows["diff"] + rows["sign"] * -80.0)
    ).astype(float)

    low, high = bootstrap_interval(rows, n_boot=300)

    assert low < -80.0 < high
    assert high - low > 10
    assert (low, high) == bootstrap_interval(rows, n_boot=300)


def test_transitions_on_a_larger_league():
    games, _ = compute_elo(with_promotion(synthetic_raw_two_divisions()))

    found = transitions(games)

    assert sorted(found.itertuples(index=False, name=None)) == [
        ("T13", "2021-22", "relegated"),
        ("U00", "2021-22", "promoted"),
    ]


def test_cli_reports_promoted_and_relegated(tmp_path, capsys):
    write_as_csvs(with_promotion(synthetic_raw_two_divisions()), tmp_path)

    assert main(["--data", str(tmp_path), "--first", "10"]) == 0

    out = capsys.readouterr().out
    assert "Ascendidos" in out and "Descendidos" in out
    assert "Desfase combinado d =" in out and "IC 95 %" in out


def test_cli_handles_data_without_any_transition(tmp_path, capsys):
    write_as_csvs(synthetic_raw_two_divisions(), tmp_path)

    assert main(["--data", str(tmp_path)]) == 0

    assert "No hay ascensos ni descensos" in capsys.readouterr().out
