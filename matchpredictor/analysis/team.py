"""Análisis del historial de un equipo a partir del DataFrame que devuelve ``load_all``.

La base es el formato largo: una fila por equipo y partido, vista desde su punto de vista
(``goals_for``, ``goals_against``, ``venue``...). Es lo que permite filtrar "un equipo" sin
tratar por separado los partidos de local y de visitante, y es también el punto de partida
de las features del modelo (forma, Elo, descanso).
"""

from __future__ import annotations

import difflib

import pandas as pd

_POINTS = {"W": 3, "D": 1, "L": 0}

TEAM_MATCH_COLUMNS = [
    "season",
    "division",
    "date",
    "team",
    "opponent",
    "venue",
    "goals_for",
    "goals_against",
    "outcome",
    "points",
]


class UnknownTeamError(ValueError):
    """El equipo pedido no aparece en los datos."""


def list_teams(
    df: pd.DataFrame, division: str | None = None, season: str | None = None
) -> list[str]:
    """Equipos que aparecen en los datos, ordenados alfabéticamente, con filtros opcionales."""
    if division is not None:
        df = df[df["division"] == division]
    if season is not None:
        df = df[df["season"] == season]
    return sorted(set(df["home_team"]) | set(df["away_team"]))


def team_matches(df: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    """Formato largo: una fila por equipo y partido, ordenada por fecha.

    ``outcome`` es W/D/L desde el punto de vista del equipo y ``points`` 3/1/0. Si se indica
    ``team`` solo se devuelven sus partidos; si no, los de todos los equipos.
    """
    home = pd.DataFrame(
        {
            "season": df["season"],
            "division": df["division"],
            "date": df["date"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "venue": "H",
            "goals_for": df["home_goals"],
            "goals_against": df["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "season": df["season"],
            "division": df["division"],
            "date": df["date"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "venue": "A",
            "goals_for": df["away_goals"],
            "goals_against": df["home_goals"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["outcome"] = "D"
    long.loc[long["goals_for"] > long["goals_against"], "outcome"] = "W"
    long.loc[long["goals_for"] < long["goals_against"], "outcome"] = "L"
    long["points"] = long["outcome"].map(_POINTS).astype("int64")

    if team is not None:
        long = long[long["team"] == resolve_team(df, team)]
    return long.sort_values(["date", "venue"], kind="stable").reset_index(drop=True)[
        TEAM_MATCH_COLUMNS
    ]


def resolve_team(df: pd.DataFrame, team: str) -> str:
    """Nombre exacto del equipo en los datos, o ``UnknownTeamError`` con sugerencias."""
    teams = list_teams(df)
    if team in teams:
        return team
    close = difflib.get_close_matches(team, teams, n=3, cutoff=0.6)
    hint = f" ¿Quizá {', '.join(close)}?" if close else ""
    raise UnknownTeamError(f"Equipo desconocido: {team!r}.{hint}")


def _tally(matches: pd.DataFrame) -> dict[str, float | int]:
    played = len(matches)
    points = int(matches["points"].sum())
    goals_for = int(matches["goals_for"].sum())
    goals_against = int(matches["goals_against"].sum())
    return {
        "played": played,
        "wins": int((matches["outcome"] == "W").sum()),
        "draws": int((matches["outcome"] == "D").sum()),
        "losses": int((matches["outcome"] == "L").sum()),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "points": points,
        "points_per_game": round(points / played, 2) if played else 0.0,
    }


def team_summary(df: pd.DataFrame, team: str, season: str | None = None) -> dict:
    """Balance global de un equipo y su desglose en casa y fuera.

    Devuelve ``{"team", "overall", "home", "away"}``; cada bloque tiene partidos, victorias,
    empates, derrotas, goles, diferencia, puntos y puntos por partido. Con ``season`` se
    limita a esa temporada.
    """
    matches = team_matches(df, team)
    if season is not None:
        matches = matches[matches["season"] == season]
    return {
        "team": matches["team"].iloc[0] if len(matches) else team,
        "overall": _tally(matches),
        "home": _tally(matches[matches["venue"] == "H"]),
        "away": _tally(matches[matches["venue"] == "A"]),
    }


def team_form(df: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    """Últimos ``n`` partidos del equipo (los más recientes al final)."""
    if n < 1:
        raise ValueError("n debe ser al menos 1")
    return team_matches(df, team).tail(n).reset_index(drop=True)


def form_string(df: pd.DataFrame, team: str, n: int = 5) -> str:
    """Forma reciente como texto, del partido más antiguo al más reciente (p. ej. ``WDLWW``)."""
    return "".join(team_form(df, team, n)["outcome"])


def team_by_season(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Una fila por temporada (y división) con el balance del equipo.

    Sirve para ver ascensos y descensos: la columna ``division`` cambia entre temporadas.
    """
    matches = team_matches(df, team)
    rows = [
        {"season": season, "division": division, **_tally(group)}
        for (season, division), group in matches.groupby(["season", "division"], sort=True)
    ]
    return pd.DataFrame(rows)
