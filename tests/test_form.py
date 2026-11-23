import numpy as np
import pandas as pd
import pytest

from matchpredictor.features import FORM_METRICS, add_form_features, current_form


def league(n_teams=6, rounds=30, seed=1, shots=True):
    """Liga aleatoria con una jornada semanal en la que cada equipo juega una vez."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(rounds):
        order = rng.permutation(n_teams)
        for home, away in zip(order[::2], order[1::2], strict=True):
            row = {
                "season": "2022-23" if r < rounds // 2 else "2023-24",
                "division": "SP1",
                "date": pd.Timestamp("2022-08-20")
                + pd.Timedelta(days=7 * r + int(rng.integers(0, 3))),
                "home_team": f"T{home}",
                "away_team": f"T{away}",
                "home_goals": int(rng.poisson(1.4)),
                "away_goals": int(rng.poisson(1.1)),
            }
            if shots:
                row |= {
                    "home_shots": float(rng.integers(5, 20)),
                    "away_shots": float(rng.integers(5, 20)),
                    "home_shots_on_target": float(rng.integers(1, 9)),
                    "away_shots_on_target": float(rng.integers(1, 9)),
                }
            rows.append(row)
    return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)


def reference(games, i, team):
    """Implementación ingenua: recorre el historial anterior del equipo."""
    date = games.loc[i, "date"]
    past = games[
        (games["date"] < date) & ((games["home_team"] == team) | (games["away_team"] == team))
    ].sort_values("date")
    pts, gd, shots, sot = [], [], [], []
    for _, m in past.iterrows():
        sign = 1 if m["home_team"] == team else -1
        goals = sign * (m["home_goals"] - m["away_goals"])
        pts.append(3 if goals > 0 else 1 if goals == 0 else 0)
        gd.append(goals)
        shots.append(sign * (m["home_shots"] - m["away_shots"]))
        sot.append(sign * (m["home_shots_on_target"] - m["away_shots_on_target"]))

    def mean(values, window, minimum):
        tail = values[-window:]
        return float(np.mean(tail)) if len(tail) >= minimum else np.nan

    rest = min((date - past["date"].iloc[-1]).days, 14) if len(past) else np.nan
    return {
        "form_pts": mean(pts, 5, 3),
        "form_gd": mean(gd, 5, 3),
        "form_shots": mean(shots, 10, 5),
        "form_sot": mean(sot, 10, 5),
        "rest_days": rest,
    }


def test_features_match_a_naive_reference_implementation():
    games = league()

    result = add_form_features(games)

    for i in range(len(games)):
        for side, team in (
            ("home", games.loc[i, "home_team"]),
            ("away", games.loc[i, "away_team"]),
        ):
            expected = reference(games, i, team)
            for metric in FORM_METRICS:
                np.testing.assert_allclose(
                    result.loc[i, f"{metric}_{side}"],
                    expected[metric],
                    equal_nan=True,
                    err_msg=f"{metric} {side} partido {i}",
                )


def test_diff_columns_are_home_minus_away():
    result = add_form_features(league())

    for metric in FORM_METRICS:
        expected = result[f"{metric}_home"] - result[f"{metric}_away"]
        pd.testing.assert_series_equal(result[f"{metric}_diff"], expected, check_names=False)


def test_a_match_never_sees_itself_or_the_future():
    games = league()
    k = 40
    altered = games.copy()
    altered.loc[k, ["home_goals", "away_goals", "home_shots", "away_shots"]] = [9, 0, 30, 1]

    before, after = add_form_features(games), add_form_features(altered)

    own = [f"{m}_{s}" for m in FORM_METRICS for s in ("home", "away", "diff")]
    pd.testing.assert_frame_equal(before.loc[:k, own], after.loc[:k, own])  # hasta él, idéntico
    teams = {games.loc[k, "home_team"], games.loc[k, "away_team"]}
    later = games.index[
        (games.index > k) & (games["home_team"].isin(teams) | games["away_team"].isin(teams))
    ]
    assert not before.loc[later, own].equals(after.loc[later, own])  # después, sí cambia


def test_first_matches_have_no_history_and_form_needs_minimum_games():
    result = add_form_features(league(rounds=8))

    first = result[result["date"] == result["date"].min()].iloc[0]
    assert pd.isna(first["form_pts_home"]) and pd.isna(first["rest_days_home"])
    second_round = result[result["date"] <= result["date"].min() + pd.Timedelta(days=13)]
    assert second_round["form_pts_home"].isna().all()  # menos de 3 partidos previos


def test_rest_days_are_capped():
    games = pd.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "division": ["SP1"] * 3,
            "date": pd.to_datetime(["2022-09-01", "2022-09-04", "2023-03-01"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "home_goals": [1, 0, 2],
            "away_goals": [0, 0, 2],
        }
    )

    result = add_form_features(games)

    assert result["rest_days_home"].tolist()[1:] == [3.0, 14.0]  # 3 días y luego tope de 14


def test_form_carries_over_between_seasons():
    games = league(rounds=12)
    result = add_form_features(games)

    last_round_of_first_season = games[games["season"] == "2022-23"]["date"].max()
    start_of_second = games[games["season"] == "2023-24"].index[0]
    assert games.loc[start_of_second, "date"] > last_round_of_first_season
    assert pd.notna(result.loc[start_of_second, "form_pts_home"])


def test_missing_shot_columns_leave_shot_features_empty():
    games = league(shots=False)

    result = add_form_features(games)

    assert result["form_shots_diff"].isna().all() and result["form_sot_home"].isna().all()
    assert result["form_pts_diff"].notna().any() and result["rest_days_diff"].notna().any()


def test_rows_and_index_are_preserved_and_input_is_untouched():
    games = league()
    original = games.copy()

    result = add_form_features(games)

    pd.testing.assert_index_equal(result.index, games.index)
    pd.testing.assert_frame_equal(games, original)


def test_duplicate_index_is_rejected():
    games = league()
    games.index = [0] * len(games)

    with pytest.raises(ValueError, match="índice"):
        add_form_features(games)


def test_current_form_equals_the_features_of_a_match_played_right_after():
    games = league()
    last = games["date"].max()
    extra = pd.DataFrame(
        [
            {
                "season": "2023-24",
                "division": "SP1",
                "date": last + pd.Timedelta(days=20),
                "home_team": f"T{h}",
                "away_team": f"T{a}",
                "home_goals": 0,
                "away_goals": 0,
                "home_shots": 10.0,
                "away_shots": 10.0,
                "home_shots_on_target": 3.0,
                "away_shots_on_target": 3.0,
            }
            for h, a in ((0, 1), (2, 3), (4, 5))
        ]
    )
    extended = pd.concat([games, extra], ignore_index=True)

    with_next = add_form_features(extended).iloc[len(games) :]
    now = current_form(games)

    for row in with_next.itertuples():
        for side, team in (("home", row.home_team), ("away", row.away_team)):
            for metric in ("form_pts", "form_gd", "form_shots", "form_sot"):
                np.testing.assert_allclose(
                    getattr(row, f"{metric}_{side}"), now.loc[team, metric], equal_nan=True
                )


def test_current_form_has_one_row_per_team_and_empty_values_without_history():
    games = league(n_teams=6, rounds=2)  # solo 2 partidos por equipo: forma de 5 sin mínimo

    now = current_form(games)

    assert now.index.tolist() == sorted(set(games["home_team"]) | set(games["away_team"]))
    assert now["form_pts"].isna().all() and now["form_shots"].isna().all()
    assert list(now.columns) == ["form_pts", "form_gd", "form_shots", "form_sot"]


def test_current_form_uses_only_the_latest_window():
    games = league(n_teams=6, rounds=30)
    team = "T0"
    mine = games[(games["home_team"] == team) | (games["away_team"] == team)]
    sot = [
        (r.home_shots_on_target - r.away_shots_on_target) * (1 if r.home_team == team else -1)
        for r in mine.itertuples()
    ]

    now = current_form(games)

    assert now.loc[team, "form_sot"] == pytest.approx(np.mean(sot[-10:]))
    assert now.loc[team, "form_sot"] != pytest.approx(np.mean(sot))
