import pandas as pd
import pytest

from matchpredictor.features import compute_elo, expected_score, goal_multiplier, latest_ratings


def _games(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "season",
            "division",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.NA
    return df


FLAT = {"initial": {}, "default_initial": 1500.0, "home_advantage": 0.0}


def test_expected_score_is_symmetric_and_monotonic():
    assert expected_score(0) == 0.5
    assert expected_score(100) == pytest.approx(1 - expected_score(-100))
    assert expected_score(200) > expected_score(100) > 0.5


@pytest.mark.parametrize(
    ("diff", "expected"), [(0, 1.0), (1, 1.0), (-2, 1.5), (3, 1.75), (-5, 2.0)]
)
def test_goal_multiplier(diff, expected):
    assert goal_multiplier(diff) == expected


def test_win_between_equals_moves_half_of_k():
    games = _games([("2024-25", "SP1", "2024-08-17", "A", "B", 1, 0)])

    _, ratings = compute_elo(games, k=20, **FLAT)

    assert ratings == {"A": 1510.0, "B": 1490.0}


def test_draw_between_equals_changes_nothing():
    games = _games([("2024-25", "SP1", "2024-08-17", "A", "B", 2, 2)])

    _, ratings = compute_elo(games, **FLAT)

    assert ratings == {"A": 1500.0, "B": 1500.0}


def test_updates_are_zero_sum():
    games = _games(
        [
            ("2024-25", "SP1", "2024-08-17", "A", "B", 3, 0),
            ("2024-25", "SP1", "2024-08-24", "C", "A", 1, 2),
            ("2024-25", "SP1", "2024-08-31", "B", "C", 0, 0),
        ]
    )

    _, ratings = compute_elo(games, **FLAT)

    assert sum(ratings.values()) == pytest.approx(4500.0)


def test_pre_match_ratings_do_not_include_the_match_itself():
    games = _games(
        [
            ("2024-25", "SP1", "2024-08-17", "A", "B", 1, 0),
            ("2024-25", "SP1", "2024-08-24", "A", "C", 1, 0),
        ]
    )

    result, _ = compute_elo(games, k=20, **FLAT)

    assert result["home_elo"].tolist() == [1500.0, 1510.0]  # la 2.ª ya ve el efecto de la 1.ª
    assert result["away_elo"].tolist() == [1500.0, 1500.0]
    assert result["elo_diff"].tolist() == [0.0, 10.0]


def test_changing_the_result_of_a_match_does_not_alter_its_own_features():
    rows = [
        ("2024-25", "SP1", "2024-08-17", "A", "B", 1, 0),
        ("2024-25", "SP1", "2024-08-24", "A", "C", 1, 0),
    ]
    altered = [*rows[:1], ("2024-25", "SP1", "2024-08-24", "A", "C", 0, 5)]

    before, _ = compute_elo(_games(rows), **FLAT)
    after, _ = compute_elo(_games(altered), **FLAT)

    assert before.loc[1, ["home_elo", "away_elo", "elo_expected_home"]].tolist() == (
        after.loc[1, ["home_elo", "away_elo", "elo_expected_home"]].tolist()
    )


def test_home_advantage_reduces_the_gain_of_a_home_win():
    games = _games([("2024-25", "SP1", "2024-08-17", "A", "B", 1, 0)])

    _, flat = compute_elo(games, k=20, **FLAT)
    _, with_advantage = compute_elo(games, k=20, home_advantage=60, initial={})

    assert with_advantage["A"] < flat["A"]
    assert compute_elo(games, home_advantage=60, initial={})[0]["elo_expected_home"].iloc[0] > 0.5


def test_big_win_moves_more_than_narrow_win():
    narrow = _games([("2024-25", "SP1", "2024-08-17", "A", "B", 1, 0)])
    big = _games([("2024-25", "SP1", "2024-08-17", "A", "B", 3, 0)])

    _, r_narrow = compute_elo(narrow, **FLAT)
    _, r_big = compute_elo(big, **FLAT)

    assert r_big["A"] - 1500 == pytest.approx(1.75 * (r_narrow["A"] - 1500))


def test_new_teams_start_by_division():
    games = _games(
        [
            ("2024-25", "SP1", "2024-08-17", "A", "B", 1, 1),
            ("2024-25", "SP2", "2024-08-17", "C", "D", 1, 1),
        ]
    )

    _, ratings = compute_elo(games, home_advantage=0.0)

    assert (ratings["A"], ratings["C"]) == (1500.0, 1400.0)


def test_rating_travels_with_the_team_across_divisions():
    games = _games(
        [
            ("2023-24", "SP2", "2023-09-01", "A", "B", 3, 0),
            ("2024-25", "SP1", "2024-09-01", "A", "C", 1, 1),
        ]
    )

    result, _ = compute_elo(games, home_advantage=0.0)

    assert result.loc[1, "home_elo"] > 1400  # A conserva lo ganado en Segunda
    assert result.loc[1, "home_elo"] == pytest.approx(1400 + 20 * 1.75 * 0.5)


def test_season_regression_pulls_ratings_towards_1500_only_at_season_change():
    rows = [
        ("2023-24", "SP1", "2023-09-01", "A", "B", 1, 0),
        ("2023-24", "SP1", "2023-09-08", "A", "C", 0, 0),
        ("2024-25", "SP1", "2024-09-01", "A", "D", 0, 0),
    ]
    flat, _ = compute_elo(_games(rows), season_regression=0.0, **FLAT)
    regressed, _ = compute_elo(_games(rows), season_regression=0.5, **FLAT)

    assert regressed.loc[1, "home_elo"] == flat.loc[1, "home_elo"]  # misma temporada
    assert 1500 < regressed.loc[2, "home_elo"] < flat.loc[2, "home_elo"]


def test_rejects_invalid_parameters():
    games = _games([("2024-25", "SP1", "2024-08-17", "A", "B", 1, 0)])
    with pytest.raises(ValueError, match="k"):
        compute_elo(games, k=0)
    with pytest.raises(ValueError, match="season_regression"):
        compute_elo(games, season_regression=1.5)


def test_latest_ratings_sorted_descending():
    series = latest_ratings({"A": 1490.0, "B": 1510.0, "C": 1500.0})

    assert series.index.tolist() == ["B", "C", "A"]


def test_division_offset_shifts_promoted_and_relegated_teams_only():
    rows = [
        ("2023-24", "SP2", "2023-09-01", "A", "B", 1, 0),  # A sube a SP1 la temporada siguiente
        ("2023-24", "SP1", "2023-09-01", "C", "D", 1, 0),  # D baja a SP2
        ("2024-25", "SP1", "2024-09-01", "A", "E", 0, 0),
        ("2024-25", "SP2", "2024-09-01", "D", "F", 0, 0),
        ("2024-25", "SP1", "2024-09-08", "C", "A", 0, 0),  # C sigue en SP1: sin ajuste
    ]
    flat = {"initial": {}, "default_initial": 1500.0, "home_advantage": 0.0}

    plain, _ = compute_elo(_games(rows), division_offset=0.0, **flat)
    shifted, _ = compute_elo(_games(rows), division_offset=-30.0, **flat)

    def rating(frame, team, season):
        game = frame[
            (frame["season"] == season)
            & ((frame["home_team"] == team) | (frame["away_team"] == team))
        ].iloc[0]
        return game["home_elo"] if game["home_team"] == team else game["away_elo"]

    assert rating(shifted, "A", "2024-25") == rating(plain, "A", "2024-25") - 30.0  # sube: + d
    assert rating(shifted, "D", "2024-25") == rating(plain, "D", "2024-25") + 30.0  # baja: - d
    assert rating(shifted, "C", "2024-25") == rating(plain, "C", "2024-25")  # se queda
    assert rating(shifted, "A", "2023-24") == rating(plain, "A", "2023-24")  # primera aparición
    assert rating(shifted, "E", "2024-25") == rating(plain, "E", "2024-25")  # equipo nuevo


def test_division_offset_default_is_neutral():
    rows = [
        ("2023-24", "SP2", "2023-09-01", "A", "B", 1, 0),
        ("2024-25", "SP1", "2024-09-01", "A", "C", 0, 0),
    ]

    default, _ = compute_elo(_games(rows))
    explicit, _ = compute_elo(_games(rows), division_offset=0.0)

    assert default.equals(explicit)
