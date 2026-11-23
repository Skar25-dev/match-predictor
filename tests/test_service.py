import numpy as np
import pandas as pd
import pytest

from matchpredictor.analysis import UnknownTeamError, team_summary
from matchpredictor.service import PredictorService
from tests.synthetic import add_shots, synthetic_raw_two_divisions


@pytest.fixture(scope="module")
def raw():
    return synthetic_raw_two_divisions()


@pytest.fixture(scope="module")
def service(raw):
    return PredictorService(raw)


def test_teams_by_division_lists_current_teams_sorted(service):
    teams = service.teams_by_division()

    assert list(teams) == ["SP1", "SP2"]
    assert teams["SP1"] == [f"T{i:02d}" for i in range(14)]
    assert teams["SP2"] == [f"U{i:02d}" for i in range(12)]


def test_empty_data_is_rejected(raw):
    with pytest.raises(ValueError, match="partidos"):
        PredictorService(raw.iloc[0:0])


def test_overview_matches_the_analysis_layer(service, raw):
    data = service.overview("T03")

    assert data["team"] == "T03"
    assert data["division"] == "SP1" and data["active"] is True
    assert data["all_time"]["overall"] == team_summary(raw, "T03")["overall"]
    latest = team_summary(raw, "T03", season=service.latest_season)
    assert data["current_season"]["overall"] == latest["overall"]
    assert len(data["form"]) == 5 and set(data["form"]) <= {"W", "D", "L"}
    assert [row["season"] for row in data["seasons"]] == sorted(raw["season"].unique())


def test_overview_last_matches_are_the_five_most_recent_newest_first(service):
    matches = service.overview("T03")["last_matches"]

    dates = [m["date"] for m in matches]
    assert len(matches) == 5
    assert dates == sorted(dates, reverse=True)
    assert set(matches[0]) == {"date", "opponent", "venue", "score", "outcome"}


def test_elo_rank_is_consistent_with_ratings(service):
    ranks = {team: service.overview(team)["elo_rank"] for team in service.current_division}

    assert sorted(ranks.values()) == list(range(1, 27))
    best = max(service.current_division, key=lambda team: service.ratings[team])
    assert ranks[best] == 1
    assert service.overview(best)["elo_rank_of"] == 26


def test_overview_unknown_team_suggests_names(service):
    with pytest.raises(UnknownTeamError, match="T03"):
        service.overview("T3")


def test_predict_returns_valid_probabilities_and_fair_odds(service):
    result = service.predict("T00", "T01")

    probs = result["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=2e-4)
    for key in probs:
        assert result["fair_odds"][key] == pytest.approx(1 / probs[key], abs=0.01)
    assert result["division"] == "SP1" and result["hypothetical"] is False


def test_predict_favours_the_stronger_team(service):
    ranked = sorted(service.current_division, key=lambda team: service.ratings[team])
    weak = next(team for team in ranked if team.startswith("T"))
    strong = next(team for team in reversed(ranked) if team.startswith("T"))

    strong_home = service.predict(strong, weak)["probabilities"]
    weak_home = service.predict(weak, strong)["probabilities"]

    assert strong_home["home"] > strong_home["away"]
    assert strong_home["home"] > weak_home["home"]
    assert weak_home["away"] > weak_home["home"]


def test_predict_across_divisions_is_flagged_hypothetical(service):
    result = service.predict("T00", "U00")

    assert result["hypothetical"] is True
    assert result["division"] == "SP1"  # por defecto, la del equipo local


def test_predict_defaults_to_the_division_of_the_home_team(service):
    assert service.predict("U00", "U01")["division"] == "SP2"
    assert service.predict("U00", "U01", division="SP1")["division"] == "SP1"


def test_predict_division_override_changes_the_probabilities(service):
    as_sp1 = service.predict("T00", "T01", division="SP1")["probabilities"]
    as_sp2 = service.predict("T00", "T01", division="SP2")["probabilities"]

    assert as_sp1 != as_sp2


def test_predict_rejects_bad_input(service):
    with pytest.raises(ValueError, match="distintos"):
        service.predict("T00", "T00")
    with pytest.raises(ValueError, match="División no válida"):
        service.predict("T00", "T01", division="SP9")
    with pytest.raises(UnknownTeamError):
        service.predict("Atlantis", "T01")


def test_team_that_no_longer_plays_still_has_a_page_and_predictions(raw):
    old = {
        "season": "2018-19",
        "division": "SP1",
        "date": pd.Timestamp("2019-05-01"),
        "time": pd.NA,
        "home_team": "Old",
        "away_team": "T00",
        "home_goals": 2,
        "away_goals": 0,
        "result": "H",
    }
    service = PredictorService(pd.concat([pd.DataFrame([old]), raw], ignore_index=True))

    data = service.overview("Old")

    assert data["active"] is False and data["elo_rank"] is None
    assert data["current_season"] is None and data["division"] == "SP1"
    assert "Old" not in service.teams_by_division()["SP1"]
    assert service.predict("Old", "T00")["probabilities"]["home"] > 0


def test_division_offset_only_affects_cross_division_predictions(raw):
    plain = PredictorService(raw, division_offset=0.0)
    shifted = PredictorService(raw, division_offset=-100.0)

    same_plain = plain.predict("T00", "T01")["probabilities"]
    same_shifted = shifted.predict("T00", "T01")["probabilities"]
    cross_plain = plain.predict("T00", "U00")
    cross_shifted = shifted.predict("T00", "U00")

    assert same_plain == same_shifted  # misma división: el desfase no interviene
    # Segunda "vale menos": el local de La Liga gana más veces contra un equipo de Segunda
    assert cross_shifted["probabilities"]["home"] > cross_plain["probabilities"]["home"]
    assert (cross_shifted["division_offset"], cross_plain["division_offset"]) == (-100.0, 0.0)
    assert shifted.predict("T00", "T01")["division_offset"] == 0.0


def test_division_offset_applies_in_both_directions(raw):
    plain = PredictorService(raw)
    shifted = PredictorService(raw, division_offset=-100.0)

    # El equipo de Segunda juega de local: con el ajuste tiene menos opciones
    assert (
        shifted.predict("U00", "T00")["probabilities"]["home"]
        < plain.predict("U00", "T00")["probabilities"]["home"]
    )


def test_ranking_uses_the_common_scale(raw):
    service = PredictorService(raw, division_offset=-1000.0)

    ranks = {team: service.overview(team)["elo_rank"] for team in service.current_division}

    assert sorted(ranks[team] for team in ranks if team.startswith("T")) == list(range(1, 15))
    assert sorted(ranks[team] for team in ranks if team.startswith("U")) == list(range(15, 27))


@pytest.fixture(scope="module")
def shots_service(raw):
    return PredictorService(add_shots(raw))


def test_overview_reports_shot_form_only_when_there_is_shot_data(service, shots_service):
    assert service.overview("T03")["sot_form"] is None  # datos sin tiros
    value = shots_service.overview("T03")["sot_form"]
    assert isinstance(value, float)
    assert value == round(float(shots_service.form.loc["T03", "form_sot"]), 1)


def test_predict_exposes_shot_form_for_both_teams(shots_service, service):
    result = shots_service.predict("T01", "T02")

    assert result["sot_form"] == {
        "home": shots_service.overview("T01")["sot_form"],
        "away": shots_service.overview("T02")["sot_form"],
    }
    assert service.predict("T01", "T02")["sot_form"] == {"home": None, "away": None}


class _Spy:
    """Modelo falso que registra las filas que recibe."""

    def __init__(self):
        self.rows = []

    def predict_proba(self, rows):
        self.rows.append(rows)
        return pd.DataFrame({"p_home": [0.5], "p_draw": [0.3], "p_away": [0.2]})


def test_same_division_uses_elo_and_shot_form_differences(raw):
    service = PredictorService(add_shots(raw))
    sot_spy, elo_spy = _Spy(), _Spy()
    service.model, service.elo_model = sot_spy, elo_spy

    result = service.predict("T01", "T02")

    assert len(sot_spy.rows) == 1 and not elo_spy.rows
    row = sot_spy.rows[0].iloc[0]
    assert row["elo_diff"] == pytest.approx(service.ratings["T01"] - service.ratings["T02"])
    expected_sot = service.overview("T01")["sot_form"] - service.overview("T02")["sot_form"]
    assert row["form_sot_diff"] == pytest.approx(expected_sot)
    assert result["model"] == "elo_sot"


def test_cross_division_uses_only_the_elo_model_in_the_common_scale(raw):
    service = PredictorService(add_shots(raw), division_offset=-40.0)
    sot_spy, elo_spy = _Spy(), _Spy()
    service.model, service.elo_model = sot_spy, elo_spy

    result = service.predict("T01", "U02")

    assert len(elo_spy.rows) == 1 and not sot_spy.rows
    row = elo_spy.rows[0].iloc[0]
    assert row["elo_diff"] == pytest.approx(
        service._common_rating("T01") - service._common_rating("U02")
    )
    assert row["division"] == "SP1"
    assert result["model"] == "elo"
    assert result["sot_form"] == {"home": None, "away": None}  # no se usan: no se muestran


def test_cross_division_probabilities_ignore_shot_form(raw):
    service = PredictorService(add_shots(raw))
    before = service.predict("T01", "U02")["probabilities"]

    service.form["form_sot"] = service.form["form_sot"] + 25.0 * np.arange(len(service.form))

    assert service.predict("T01", "U02")["probabilities"] == before


def test_trained_model_uses_elo_and_shot_form_plus_division(shots_service):
    model = shots_service.model

    assert model.features == ["elo_diff", "form_sot_diff"]
    assert model.model_.coef_.shape[1] == 3  # Elo, tiros a puerta y división
