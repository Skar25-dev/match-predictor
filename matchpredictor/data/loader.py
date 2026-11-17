"""Carga y limpieza de los CSV de football-data.co.uk.

Cada CSV contiene una temporada de una división (SP1 = La Liga, SP2 = Segunda).
Este módulo los convierte en un DataFrame uniforme: nombres en snake_case, tipos
correctos, solo partidos ya jugados y únicamente las columnas que usa el proyecto
(los archivos originales traen más de 100, la mayoría cuotas de casas de apuestas).

Los archivos de temporadas antiguas tienen menos columnas y fechas con año de dos
dígitos; el cargador lo tolera: las columnas opcionales que falten se rellenan con NaN.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


class DataFormatError(ValueError):
    """El CSV no tiene el formato esperado de football-data.co.uk."""


class DataQualityWarning(UserWarning):
    """El CSV se pudo cargar, pero contenía datos dudosos que se han corregido o descartado."""


# Columnas obligatorias: sin ellas el archivo no sirve.
_REQUIRED = {
    "Div": "division",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
}

# Columnas opcionales: si faltan (temporadas antiguas) se rellenan con NaN.
_STATS = {
    "Time": "time",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# Cuotas 1X2: apertura (sin sufijo) y cierre (sufijo C) de Bet365, media del mercado y Pinnacle.
_ODDS_BOOKS = {
    "B365": "b365",
    "Avg": "avg",
    "PS": "ps",
    "B365C": "b365c",
    "AvgC": "avgc",
    "PSC": "psc",
}
_ODDS = {
    f"{source}{suffix}": f"{name}_{outcome}"
    for source, name in _ODDS_BOOKS.items()
    for suffix, outcome in (("H", "home"), ("D", "draw"), ("A", "away"))
}

_OPTIONAL = {**_STATS, **_ODDS}
_NUMERIC = [name for name in _OPTIONAL.values() if name != "time"]

_COLUMN_ORDER = [
    "season",
    "division",
    "date",
    "time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]
_COLUMN_ORDER += [name for name in _OPTIONAL.values() if name != "time"]

_KEY = ["date", "home_team", "away_team"]


def parse_dates(values: pd.Series) -> pd.Series:
    """Convierte fechas ``dd/mm/yyyy`` o ``dd/mm/yy`` (temporadas antiguas) a datetime.

    Lo que no se pueda interpretar queda como NaT.
    """
    text = values.astype("string").str.strip()
    long_year = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    short_year = pd.to_datetime(text, format="%d/%m/%y", errors="coerce")
    return long_year.fillna(short_year)


def season_label(dates: pd.Series) -> pd.Series:
    """Etiqueta de temporada (p. ej. ``2025-26``) a partir de la fecha de cada partido.

    La temporada empieza en agosto. Es solo una aproximación: la 2019-20 se alargó hasta
    agosto de 2020 (COVID), así que ``load_season`` prefiere la temporada del nombre del
    archivo (``SP2_2019-20.csv``) y usa esta función solo si el nombre no la indica.
    """
    start_year = np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1)
    return pd.Series(
        [f"{year}-{(year + 1) % 100:02d}" for year in start_year],
        index=dates.index,
    )


_SEASON_IN_NAME = re.compile(r"(\d{4})-(\d{2})(?!\d)")


def season_from_filename(path: str | Path) -> str | None:
    """Temporada indicada en el nombre del archivo (``2019-20``), o ``None`` si no hay."""
    match = _SEASON_IN_NAME.search(Path(path).stem)
    if match is None:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return match.group(0) if (start + 1) % 100 == end else None


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    except pd.errors.EmptyDataError as error:
        raise DataFormatError(f"{path.name}: el archivo está vacío") from error
    except pd.errors.ParserError as error:
        raise DataFormatError(f"{path.name}: no es un CSV válido ({error})") from error


def _drop_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    duplicated = df.duplicated(_KEY, keep="first")
    if duplicated.any():
        warnings.warn(
            f"{label}: {int(duplicated.sum())} partido(s) duplicado(s) descartado(s)",
            DataQualityWarning,
            stacklevel=3,
        )
        df = df[~duplicated]
    return df


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "time"], kind="stable").reset_index(drop=True)


def load_season(path: str | Path) -> pd.DataFrame:
    """Carga un CSV de una temporada y devuelve solo los partidos jugados, ya limpios."""
    path = Path(path)
    raw = _read_csv(path).dropna(how="all")

    missing = [column for column in _REQUIRED if column not in raw.columns]
    if missing:
        raise DataFormatError(f"{path.name}: faltan columnas obligatorias: {', '.join(missing)}")

    columns = {**_REQUIRED, **_OPTIONAL}
    df = raw.reindex(columns=list(columns)).rename(columns=columns)

    for column in ("division", "home_team", "away_team", "time"):
        df[column] = df[column].astype("string").str.strip()
    df["date"] = parse_dates(df["date"])
    for column in ("home_goals", "away_goals", *_NUMERIC):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Los partidos sin resultado (aún sin jugar) o sin equipos/fecha no sirven para entrenar.
    df = df.dropna(subset=["home_team", "away_team", "date", "home_goals", "away_goals"])
    df = df.astype({"home_goals": "int64", "away_goals": "int64"})

    # El resultado se calcula a partir de los goles; la columna FTR del CSV solo se usa
    # para detectar incoherencias.
    df["result"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] < df["away_goals"]],
        ["H", "A"],
        default="D",
    )
    if "FTR" in raw.columns:
        declared = raw["FTR"].reindex(df.index).astype("string").str.strip()
        mismatch = declared.notna() & (declared != df["result"])
        if mismatch.any():
            warnings.warn(
                f"{path.name}: {int(mismatch.sum())} partido(s) con FTR distinto del resultado "
                "calculado por goles; se usa el calculado",
                DataQualityWarning,
                stacklevel=2,
            )

    named_season = season_from_filename(path)
    df["season"] = named_season if named_season else season_label(df["date"])
    df = _drop_duplicates(df, path.name)
    return _sort(df)[_COLUMN_ORDER]


def load_all(data_dir: str | Path) -> pd.DataFrame:
    """Carga y une todos los CSV de un directorio, ordenados cronológicamente."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No hay CSV en {data_dir}. Descárgalos con: python -m matchpredictor.data.download"
        )
    df = pd.concat([load_season(file) for file in files], ignore_index=True)
    return _sort(_drop_duplicates(df, str(data_dir)))
