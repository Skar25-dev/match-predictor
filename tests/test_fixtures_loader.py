import pandas as pd
import pytest

from matchpredictor.data import DataFormatError, DataQualityWarning, load_fixtures
from tests.conftest import FIXTURES

SAMPLE = FIXTURES / "fixtures_sample.csv"
HEADER = "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A"


def test_real_sample_loads_with_clean_columns():
    df = load_fixtures(SAMPLE)

    assert len(df) == 12
    assert list(df.columns[:5]) == ["division", "date", "time", "home_team", "away_team"]
    bilbao = df[df["home_team"] == "Ath Bilbao"].iloc[0]
    assert bilbao["away_team"] == "Espanol" and bilbao["division"] == "SP1"
    assert bilbao["date"] == pd.Timestamp("2026-11-20") and bilbao["time"] == "20:00"
    assert (bilbao["b365_home"], bilbao["b365_draw"], bilbao["b365_away"]) == (1.5, 4.33, 6.5)
    assert bilbao["avg_home"] == pytest.approx(1.48)
    first = df.iloc[0]  # el primero por fecha y hora es el de Segunda de las 19:30
    assert (first["home_team"], first["time"]) == ("Andorra", "19:30")
    assert df["b365c_home"].isna().all() and df["avgc_home"].isna().all()  # sin cuotas de cierre


def test_fixtures_are_sorted_by_date_and_time():
    df = load_fixtures(SAMPLE)

    assert df["date"].is_monotonic_increasing
    same_day = df[df["date"] == pd.Timestamp("2026-11-21")]
    assert same_day["time"].tolist() == sorted(same_day["time"].tolist())


def test_only_requested_divisions_are_kept(write_csv):
    path = write_csv(
        "SP1,20/11/2026,20:00,A,B,2,3,3\nE0,20/11/2026,15:00,C,D,2,3,3\nSP2,21/11/2026,13:00,E,F,2,3,3\n",
        header=HEADER,
    )

    assert load_fixtures(path)["division"].tolist() == ["SP1", "SP2"]
    assert load_fixtures(path, divisions=("E0",))["home_team"].tolist() == ["C"]


def test_rows_without_teams_or_date_and_blank_lines_are_dropped(write_csv):
    path = write_csv(
        "SP1,20/11/2026,20:00,A,B,2,3,3\n,,,,,,,\nSP1,,20:00,C,D,2,3,3\nSP1,21/11/2026,,,E,2,3,3\n",
        header=HEADER,
    )

    assert load_fixtures(path)["home_team"].tolist() == ["A"]


def test_missing_optional_columns_become_nan(write_csv):
    path = write_csv("SP1,20/11/2026,A,B\n", header="Div,Date,HomeTeam,AwayTeam")

    df = load_fixtures(path)

    assert pd.isna(df.loc[0, "time"]) and df["b365_home"].isna().all()


def test_missing_required_columns_raise_a_clear_error(write_csv):
    path = write_csv("SP1,20/11/2026,A\n", header="Div,Date,HomeTeam")

    with pytest.raises(DataFormatError, match="AwayTeam"):
        load_fixtures(path)


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "fixtures.csv"
    empty.write_text("")

    with pytest.raises(DataFormatError, match="vacío"):
        load_fixtures(empty)


def test_duplicates_are_dropped_with_a_warning(write_csv):
    path = write_csv(
        "SP1,20/11/2026,20:00,A,B,2,3,3\nSP1,20/11/2026,20:00,A,B,2,3,3\n", header=HEADER
    )

    with pytest.warns(DataQualityWarning):
        df = load_fixtures(path)

    assert len(df) == 1


def test_whitespace_in_names_is_trimmed(write_csv):
    path = write_csv("SP1,20/11/2026,20:00, Cadiz ,Sevilla ,2,3,3\n", header=HEADER)

    row = load_fixtures(path).iloc[0]

    assert (row["home_team"], row["away_team"]) == ("Cadiz", "Sevilla")


def test_load_all_ignores_fixture_files_placed_next_to_the_results(tmp_path):
    import shutil

    from matchpredictor.data import load_all

    for name in ("SP1_2025-26_sample.csv", "SP2_2025-26_sample.csv"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    shutil.copy(SAMPLE, tmp_path / "fixtures.csv")  # el calendario NO son resultados

    games = load_all(tmp_path)

    assert len(games) == 19  # 11 de SP1 + 8 de SP2; nada del calendario
    assert games["season"].unique().tolist() == ["2025-26"]
