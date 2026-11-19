import pandas as pd
import pytest

from matchpredictor.evaluation.tuning import (
    DEFAULT_PARAMS,
    edge_parameters,
    elo_games,
    grid_search,
    main,
    score_params,
)
from tests.synthetic import synthetic_raw, write_as_csvs

VALIDATION = ["2021-22", "2022-23"]


@pytest.fixture(scope="module")
def raw():
    return synthetic_raw()


def test_sp2_initial_sets_the_starting_rating_of_segunda_teams():
    base = {
        "season": "2024-25",
        "time": pd.NA,
        "home_goals": 1,
        "away_goals": 1,
        "result": "D",
    }
    raw = pd.DataFrame(
        [
            {
                **base,
                "division": "SP1",
                "date": pd.Timestamp("2024-08-17"),
                "home_team": "A",
                "away_team": "B",
            },
            {
                **base,
                "division": "SP2",
                "date": pd.Timestamp("2024-08-17"),
                "home_team": "C",
                "away_team": "D",
            },
        ]
    )

    games = elo_games(raw, {"sp2_initial": 1350.0})

    by_team = games.set_index("home_team")["home_elo"]
    assert (by_team["A"], by_team["C"]) == (1500.0, 1350.0)


def test_every_parameter_reaches_compute_elo(raw):
    base = elo_games(raw, {})

    for change in ({"k": 40.0}, {"home_advantage": 0.0}, {"season_regression": 0.5}):
        changed = elo_games(raw, change)
        assert not changed["home_elo"].equals(base["home_elo"]), change

    # Con los valores por defecto explícitos el resultado es idéntico
    assert elo_games(raw, dict(DEFAULT_PARAMS))["home_elo"].equals(base["home_elo"])


def test_unknown_parameter_is_rejected(raw):
    with pytest.raises(ValueError, match="desconocidos"):
        elo_games(raw, {"kk": 10})


def test_grid_search_has_one_row_per_combination_sorted_best_first(raw):
    grid = {"k": [5, 20, 60], "home_advantage": [60]}

    results = grid_search(raw, grid, VALIDATION)

    assert len(results) == 3
    assert results["log_loss"].is_monotonic_increasing
    assert {"k", "home_advantage", "log_loss"} <= set(results.columns)


def test_grid_search_prefers_a_sensible_k_on_stable_strengths(raw):
    """Las fuerzas sintéticas son constantes: un k enorme solo añade ruido."""
    results = grid_search(raw, {"k": [5, 20, 120]}, VALIDATION).set_index("k")["log_loss"]

    assert results[120] > results[20]
    assert results[120] > results[5]


def test_score_params_matches_grid_search_row(raw):
    direct = score_params(raw, {"k": 30.0}, VALIDATION)

    row = grid_search(raw, {"k": [30.0]}, VALIDATION).iloc[0]

    assert row["log_loss"] == pytest.approx(direct)


def test_edge_parameters_flags_values_on_the_border_of_the_grid():
    grid = {"k": [10, 20, 30], "home_advantage": [60], "season_regression": [0.0, 0.2, 0.4]}
    best = {"k": 20, "home_advantage": 60, "season_regression": 0.4}

    assert edge_parameters(best, grid) == ["season_regression"]  # k interior; 1 valor no cuenta


def test_cli_runs_end_to_end(tmp_path, capsys, raw):
    write_as_csvs(raw, tmp_path)

    assert main(["--data", str(tmp_path), "--holdout", "2", "--quick"]) == 0

    out = capsys.readouterr().out
    assert "Validación: 2021-22" in out and "Holdout: 2022-23, 2023-24" in out
    for model in ("frecuencias", "elo_actual", "elo_ajustado", "market"):
        assert model in out
    assert "MEJORES COMBINACIONES" in out


def test_cli_fails_clearly_without_validation_seasons(tmp_path, raw):
    write_as_csvs(raw, tmp_path)

    with pytest.raises(SystemExit, match="validación"):
        main(["--data", str(tmp_path), "--holdout", "5", "--quick"])
