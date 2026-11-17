import pandas as pd
import pytest

from matchpredictor.analysis import (
    UnknownTeamError,
    form_string,
    list_teams,
    team_by_season,
    team_form,
    team_matches,
    team_summary,
)


def _matches():
    rows = [
        # season, division, date, home, away, hg, ag
        ("2023-24", "SP1", "2023-08-12", "Cadiz", "Sevilla", 1, 1),
        ("2023-24", "SP1", "2023-08-19", "Real Madrid", "Cadiz", 3, 0),
        ("2023-24", "SP1", "2023-08-26", "Cadiz", "Betis", 2, 0),
        ("2023-24", "SP1", "2023-09-02", "Sevilla", "Real Madrid", 0, 2),
        ("2024-25", "SP2", "2024-08-17", "Cadiz", "Eibar", 0, 1),
        ("2024-25", "SP2", "2024-08-24", "Levante", "Cadiz", 1, 2),
    ]
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
    return df


def test_list_teams_sorted_and_filtered():
    df = _matches()

    assert list_teams(df) == ["Betis", "Cadiz", "Eibar", "Levante", "Real Madrid", "Sevilla"]
    assert list_teams(df, division="SP2") == ["Cadiz", "Eibar", "Levante"]
    assert list_teams(df, season="2023-24", division="SP1") == [
        "Betis",
        "Cadiz",
        "Real Madrid",
        "Sevilla",
    ]


def test_team_matches_has_two_rows_per_match_and_mirrors_goals():
    long = team_matches(_matches())

    assert len(long) == 12
    first = long[(long["team"] == "Real Madrid") & (long["opponent"] == "Cadiz")].iloc[0]
    assert (first["venue"], first["goals_for"], first["goals_against"]) == ("H", 3, 0)
    assert (first["outcome"], first["points"]) == ("W", 3)
    mirror = long[(long["team"] == "Cadiz") & (long["opponent"] == "Real Madrid")].iloc[0]
    assert (mirror["venue"], mirror["goals_for"], mirror["goals_against"]) == ("A", 0, 3)
    assert (mirror["outcome"], mirror["points"]) == ("L", 0)


def test_team_matches_filters_one_team_in_date_order():
    long = team_matches(_matches(), "Cadiz")

    assert len(long) == 5
    assert set(long["team"]) == {"Cadiz"}
    assert long["date"].is_monotonic_increasing
    assert long["opponent"].tolist() == ["Sevilla", "Real Madrid", "Betis", "Eibar", "Levante"]
    assert long["outcome"].tolist() == ["D", "L", "W", "L", "W"]


def test_team_summary_overall_home_away():
    summary = team_summary(_matches(), "Cadiz")

    # Cadiz: E 1-1 (casa), D 0-3 (fuera), V 2-0 (casa), D 0-1 (casa), V 2-1 (fuera)
    overall = summary["overall"]
    assert overall["played"] == 5
    assert (overall["wins"], overall["draws"], overall["losses"]) == (2, 1, 2)
    assert (overall["goals_for"], overall["goals_against"]) == (5, 6)
    assert overall["goal_difference"] == -1
    assert overall["points"] == 7
    assert overall["points_per_game"] == 1.4
    assert (summary["home"]["played"], summary["home"]["points"]) == (3, 4)
    assert (summary["away"]["played"], summary["away"]["points"]) == (2, 3)


def test_team_summary_for_a_single_season():
    summary = team_summary(_matches(), "Cadiz", season="2024-25")

    assert summary["overall"]["played"] == 2
    assert summary["overall"]["wins"] == 1
    assert summary["overall"]["points"] == 3


def test_form_returns_latest_matches_oldest_first():
    df = _matches()

    form = team_form(df, "Cadiz", n=3)

    assert form["opponent"].tolist() == ["Betis", "Eibar", "Levante"]
    assert form_string(df, "Cadiz", n=3) == "WLW"
    assert form_string(df, "Cadiz", n=100) == "DLWLW"  # n mayor que el historial: devuelve todo


def test_form_rejects_invalid_n():
    with pytest.raises(ValueError, match="al menos 1"):
        team_form(_matches(), "Cadiz", n=0)


def test_team_by_season_shows_division_change():
    table = team_by_season(_matches(), "Cadiz")

    assert table["season"].tolist() == ["2023-24", "2024-25"]
    assert table["division"].tolist() == ["SP1", "SP2"]
    assert table["played"].tolist() == [3, 2]


def test_unknown_team_suggests_close_names():
    with pytest.raises(UnknownTeamError, match="Real Madrid"):
        team_matches(_matches(), "Real Madri")
    with pytest.raises(UnknownTeamError, match="desconocido"):
        team_summary(_matches(), "Atlantis FC")
