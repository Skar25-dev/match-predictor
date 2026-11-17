import shutil

import pandas as pd
import pytest

from matchpredictor.data import (
    DataFormatError,
    DataQualityWarning,
    load_all,
    load_season,
    parse_dates,
    season_from_filename,
    season_label,
)
from tests.conftest import FIXTURES


# --- Muestras reales (recortadas de los CSV de football-data.co.uk, temporada 2025-26) ---------
def test_real_sp1_sample_loads_clean():
    df = load_season(FIXTURES / "SP1_2025-26_sample.csv")

    assert len(df) == 11
    assert list(df.columns[:9]) == [
        "season", "division", "date", "time", "home_team", "away_team",
        "home_goals", "away_goals", "result",
    ]  # fmt: skip
    assert set(df["division"]) == {"SP1"}
    assert set(df["season"]) == {"2025-26"}
    assert set(df["result"]) <= {"H", "D", "A"}
    assert df["home_goals"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing
    assert {"home_shots", "b365_home", "avgc_home", "psc_away"} <= set(df.columns)
    assert "Div" not in df.columns  # las columnas originales se renombran


def test_real_sp2_sample_contains_cadiz():
    df = load_season(FIXTURES / "SP2_2025-26_sample.csv")

    assert set(df["division"]) == {"SP2"}
    assert "Cadiz" in set(df["home_team"]) | set(df["away_team"])


def test_real_sample_odds_are_numeric_and_valid():
    df = load_season(FIXTURES / "SP1_2025-26_sample.csv")

    for column in ("b365c_home", "b365c_draw", "b365c_away"):
        assert pd.api.types.is_float_dtype(df[column])
        assert (df[column] > 1).all()  # una cuota decimal siempre es mayor que 1


# --- Fechas y temporadas -----------------------------------------------------------------------
def test_parse_dates_accepts_four_and_two_digit_years():
    parsed = parse_dates(pd.Series(["15/08/2025", "15/08/25", "no es una fecha"]))

    assert parsed.iloc[0] == pd.Timestamp("2025-08-15")
    assert parsed.iloc[1] == pd.Timestamp("2025-08-15")
    assert pd.isna(parsed.iloc[2])


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2020-07-19", "2019-20"),  # la 2019-20 acabó en julio por el COVID
        ("2020-08-15", "2020-21"),
        ("2021-01-10", "2020-21"),
        ("2021-05-23", "2020-21"),
        ("1999-09-01", "1999-00"),
    ],
)
def test_season_label(date, expected):
    assert season_label(pd.Series([pd.Timestamp(date)])).iloc[0] == expected


# --- Limpieza ----------------------------------------------------------------------------------
def test_result_is_derived_from_goals(write_csv):
    path = write_csv("SP1,19/08/2025,Alpha,Beta,2,1,H\nSP1,20/08/2025,Gamma,Delta,0,3,A\n"
                     "SP1,21/08/2025,Eps,Zeta,1,1,D\n")  # fmt: skip

    assert load_season(path)["result"].tolist() == ["H", "A", "D"]


def test_inconsistent_ftr_warns_and_uses_goals(write_csv):
    path = write_csv("SP1,19/08/2025,Alpha,Beta,2,1,A\n")  # 2-1 pero FTR dice visitante

    with pytest.warns(DataQualityWarning, match="FTR"):
        df = load_season(path)

    assert df["result"].tolist() == ["H"]


def test_old_format_two_digit_years_and_missing_optional_columns(write_csv):
    path = write_csv("SP1,19/08/16,Alpha,Beta,2,1,H\nSP1,27/05/17,Gamma,Delta,0,0,D\n")

    df = load_season(path)

    assert df["date"].tolist() == [pd.Timestamp("2016-08-19"), pd.Timestamp("2017-05-27")]
    assert set(df["season"]) == {"2016-17"}
    assert df["time"].isna().all()
    assert df["b365c_home"].isna().all()  # la columna existe aunque el CSV no la traiga
    assert df["home_shots"].isna().all()


def test_unplayed_and_blank_rows_are_dropped(write_csv):
    path = write_csv("SP1,19/08/2025,Alpha,Beta,2,1,H\nSP1,26/08/2025,Gamma,Delta,,,\n,,,,,,\n")

    df = load_season(path)

    assert df["home_team"].tolist() == ["Alpha"]


def test_duplicate_matches_are_dropped_with_warning(write_csv):
    path = write_csv("SP1,19/08/2025,Alpha,Beta,2,1,H\nSP1,19/08/2025,Alpha,Beta,2,1,H\n")

    with pytest.warns(DataQualityWarning, match="duplicado"):
        df = load_season(path)

    assert len(df) == 1


def test_team_names_are_stripped(write_csv):
    path = write_csv("SP1,19/08/2025, Alpha ,Beta  ,2,1,H\n")

    df = load_season(path)

    assert df["home_team"].tolist() == ["Alpha"]
    assert df["away_team"].tolist() == ["Beta"]


def test_latin1_files_are_read(write_csv):
    path = write_csv("SP2,19/08/2005,Málaga,Alpha,1,0,H\n", encoding="latin-1")

    assert load_season(path)["home_team"].tolist() == ["Málaga"]


def test_missing_required_column_raises(write_csv):
    path = write_csv("SP1,19/08/2025,Alpha,Beta,H\n", header="Div,Date,HomeTeam,AwayTeam,FTR")

    with pytest.raises(DataFormatError, match="FTHG"):
        load_season(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")

    with pytest.raises(DataFormatError, match="vacío"):
        load_season(path)


# --- load_all ----------------------------------------------------------------------------------
def test_load_all_concatenates_and_sorts_chronologically(tmp_path):
    for sample in FIXTURES.glob("*.csv"):
        shutil.copy(sample, tmp_path / sample.name)

    df = load_all(tmp_path)

    assert len(df) == 11 + 8
    assert set(df["division"]) == {"SP1", "SP2"}
    assert df["date"].is_monotonic_increasing
    assert df.index.tolist() == list(range(len(df)))


def test_load_all_drops_matches_repeated_across_files(tmp_path):
    sample = FIXTURES / "SP1_2025-26_sample.csv"
    shutil.copy(sample, tmp_path / "SP1_a.csv")
    shutil.copy(sample, tmp_path / "SP1_b.csv")

    with pytest.warns(DataQualityWarning, match="duplicado"):
        df = load_all(tmp_path)

    assert len(df) == 11


def test_load_all_without_csv_files_explains_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_all(tmp_path)


def test_season_comes_from_filename_for_matches_played_after_august(write_csv):
    """Caso real: Fuenlabrada-Deportivo (2019-20) se jugó en agosto de 2020 por el COVID."""
    path = write_csv(
        "SP2,20/06/20,Alpha,Beta,1,0,H\nSP2,10/08/20,Fuenlabrada,La Coruna,1,0,H\n",
        name="SP2_2019-20.csv",
    )

    assert set(load_season(path)["season"]) == {"2019-20"}


def test_season_falls_back_to_dates_when_filename_has_none(write_csv):
    path = write_csv("SP2,10/08/20,Alpha,Beta,1,0,H\n", name="season.csv")

    assert set(load_season(path)["season"]) == {"2020-21"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("SP2_2019-20.csv", "2019-20"),
        ("SP1_1999-00.csv", "1999-00"),
        ("season.csv", None),
        ("SP1_2019-25.csv", None),  # años no consecutivos: no es una temporada válida
    ],
)
def test_season_from_filename(name, expected):
    assert season_from_filename(name) == expected
