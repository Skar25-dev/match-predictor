"""Ratings Elo que se arrastran entre temporadas y divisiones.

Cádiz (Segunda) y Sevilla / Real Madrid (La Liga) casi nunca se enfrentan en una misma
temporada, así que no se pueden comparar por sus resultados directos. Un Elo continuo sí:
cada equipo conserva su rating al ascender o descender, y los partidos entre equipos de
distinta división no existen, pero los ascendidos y descendidos conectan los dos grupos.

Los ratings que se devuelven para cada partido son **previos** al partido (sin fuga de
información), listos para usarse como features.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_INITIAL = {"SP1": 1500.0, "SP2": 1400.0}
BASE_RATING = 1500.0


def expected_score(rating_diff: float) -> float:
    """Resultado esperado (0-1) de un equipo con ``rating_diff`` puntos más que el rival."""
    return 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))


def goal_multiplier(goal_difference: int) -> float:
    """Peso extra de las goleadas (fórmula del World Football Elo)."""
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def compute_elo(
    df: pd.DataFrame,
    k: float = 20.0,
    home_advantage: float = 60.0,
    initial: dict[str, float] | None = None,
    default_initial: float = BASE_RATING,
    season_regression: float = 0.0,
    division_offset: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Recorre los partidos en orden cronológico y calcula el Elo de cada equipo.

    Devuelve ``(partidos, ratings)``: los partidos (ordenados por fecha) con las columnas
    ``home_elo``, ``away_elo``, ``elo_diff`` (local menos visitante, sin ventaja de campo) y
    ``elo_expected_home`` (resultado esperado del local con ventaja de campo), todas **previas**
    al partido; y los ratings actuales de cada equipo tras el último partido.

    - ``k``: cuánto se mueve un rating por partido.
    - ``home_advantage``: puntos extra del local al calcular el resultado esperado.
    - ``initial``: rating de un equipo la primera vez que aparece, según su división
      (por defecto los de Segunda empiezan por debajo que los de La Liga).
    - ``season_regression``: fracción (0-1) con la que, al empezar una temporada nueva, cada
      rating se acerca a 1500. Sirve para reflejar plantillas cambiantes.
    - ``division_offset``: *un rating de Segunda equivale a (rating + offset) en La Liga*. Al
      ascender se suma y al descender se resta, para que el rating esté en la escala de la
      división en la que juega. Con un valor negativo, Segunda "vale menos" que La Liga. Se
      estima con ``matchpredictor.evaluation.divisions``.
    """
    if k <= 0:
        raise ValueError("k debe ser positivo")
    if not 0 <= season_regression <= 1:
        raise ValueError("season_regression debe estar entre 0 y 1")
    initial = DEFAULT_INITIAL if initial is None else initial

    games = df.sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, str] = {}
    last_division: dict[str, str] = {}
    pre_home, pre_away, expected = [], [], []

    for row in games.itertuples(index=False):
        for team in (row.home_team, row.away_team):
            if team not in ratings:
                ratings[team] = initial.get(row.division, default_initial)
            else:
                if last_season[team] != row.season and season_regression:
                    ratings[team] += season_regression * (BASE_RATING - ratings[team])
                if (last_division[team], row.division) == ("SP2", "SP1"):
                    ratings[team] += division_offset
                elif (last_division[team], row.division) == ("SP1", "SP2"):
                    ratings[team] -= division_offset
            last_season[team] = row.season
            last_division[team] = row.division

        home, away = ratings[row.home_team], ratings[row.away_team]
        expected_home = expected_score(home + home_advantage - away)
        pre_home.append(home)
        pre_away.append(away)
        expected.append(expected_home)

        if row.home_goals > row.away_goals:
            actual = 1.0
        elif row.home_goals < row.away_goals:
            actual = 0.0
        else:
            actual = 0.5
        change = k * goal_multiplier(row.home_goals - row.away_goals) * (actual - expected_home)
        ratings[row.home_team] = home + change
        ratings[row.away_team] = away - change

    games["home_elo"] = pre_home
    games["away_elo"] = pre_away
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    games["elo_expected_home"] = expected
    return games, ratings


def latest_ratings(ratings: dict[str, float]) -> pd.Series:
    """Ratings actuales ordenados de mayor a menor."""
    return pd.Series(ratings, name="elo").sort_values(ascending=False)
