"""Features de forma reciente, tiros y descanso, calculadas **antes** de cada partido.

Para cada equipo se mira su historial anterior (en cualquier división y temporada, así que al
empezar una temporada se arrastra la forma de la anterior). Nunca se usa el propio partido ni
los posteriores: cada ventana se desplaza un partido antes de promediar.

Por cada métrica ``m`` se añaden ``m_home``, ``m_away`` y ``m_diff`` (local menos visitante):

- ``form_pts``: puntos por partido en los últimos 5 (hacen falta al menos 3).
- ``form_gd``: diferencia de goles media en los últimos 5 (al menos 3).
- ``form_shots``: diferencia de tiros media en los últimos 10 (al menos 5).
- ``form_sot``: diferencia de tiros a puerta media en los últimos 10 (al menos 5).
- ``rest_days``: días desde su partido anterior (máximo 14; vacío en el primero).

Si faltan las columnas de tiros (formatos antiguos), esas métricas quedan vacías.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REST_CAP = 14
# (nombre, columna de origen en el formato largo, ventana, mínimo de partidos)
_ROLLING = (
    ("form_pts", "pts", 5, 3),
    ("form_gd", "gd", 5, 3),
    ("form_shots", "shots", 10, 5),
    ("form_sot", "sot", 10, 5),
)
FORM_METRICS = [name for name, *_ in _ROLLING] + ["rest_days"]
FORM_DIFF_FEATURES = [f"{name}_diff" for name in FORM_METRICS]


def _column(games: pd.DataFrame, name: str) -> pd.Series:
    if name in games.columns:
        return games[name].astype(float)
    return pd.Series(np.nan, index=games.index)


def add_form_features(games: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de ``games`` con las features de forma, tiros y descanso."""
    if not games.index.is_unique:
        raise ValueError("games debe tener un índice sin duplicados (usa reset_index)")

    home_goals, away_goals = games["home_goals"], games["away_goals"]
    shots_diff = _column(games, "home_shots") - _column(games, "away_shots")
    sot_diff = _column(games, "home_shots_on_target") - _column(games, "away_shots_on_target")

    def side(name: str, own: str, points: pd.Series, sign: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game": games.index,
                "side": name,
                "team": games[own].to_numpy(),
                "date": games["date"].to_numpy(),
                "pts": points.to_numpy(),
                "gd": (sign * (home_goals - away_goals)).to_numpy(),
                "shots": (sign * shots_diff).to_numpy(),
                "sot": (sign * sot_diff).to_numpy(),
            }
        )

    home_points = pd.Series(
        np.select([home_goals > away_goals, home_goals == away_goals], [3, 1], 0)
    )
    away_points = pd.Series(
        np.select([away_goals > home_goals, home_goals == away_goals], [3, 1], 0)
    )
    long = pd.concat(
        [
            side("home", "home_team", home_points, 1),
            side("away", "away_team", away_points, -1),
        ],
        ignore_index=True,
    )
    long = long.sort_values(["team", "date"], kind="stable")
    by_team = long.groupby("team", sort=False)
    for name, source, window, minimum in _ROLLING:
        long[name] = by_team[source].transform(
            lambda values, w=window, m=minimum: values.shift(1).rolling(w, min_periods=m).mean()
        )
    long["rest_days"] = by_team["date"].diff().dt.days.clip(upper=REST_CAP)

    result = games.copy()
    for side_name in ("home", "away"):
        part = long[long["side"] == side_name].set_index("game")
        for metric in FORM_METRICS:
            result[f"{metric}_{side_name}"] = part[metric].reindex(result.index)
    for metric in FORM_METRICS:
        result[f"{metric}_diff"] = result[f"{metric}_home"] - result[f"{metric}_away"]
    return result
