"""Servicio que junta datos, Elo y modelo, y responde lo que necesita la web.

Se construye una vez al arrancar (``PredictorService(load_all(...))``) y después responde sin
volver a leer ni a entrenar. Devuelve diccionarios y listas simples, listos para plantillas o
JSON.
"""

from __future__ import annotations

import pandas as pd

from matchpredictor.analysis import (
    UnknownTeamError,
    form_string,
    resolve_team,
    team_by_season,
    team_matches,
    team_summary,
)
from matchpredictor.features import compute_elo
from matchpredictor.models import EloLogit

DIVISIONS = ("SP1", "SP2")


class PredictorService:
    def __init__(self, raw: pd.DataFrame):
        if raw.empty:
            raise ValueError("No hay partidos con los que trabajar")
        self.raw = raw
        self.games, self.ratings = compute_elo(raw)
        self.model = EloLogit().fit(self.games)
        self.latest_season = str(raw["season"].max())
        self._long = team_matches(raw)
        in_season = self._long[self._long["season"] == self.latest_season]
        #: división actual de cada equipo que juega en la temporada más reciente
        self.current_division = in_season.groupby("team")["division"].last().to_dict()
        self._rank = {
            team: position
            for position, team in enumerate(
                sorted(self.current_division, key=lambda t: -self.ratings[t]), start=1
            )
        }

    # --- equipos -----------------------------------------------------------------------
    def teams_by_division(self) -> dict[str, list[str]]:
        """Equipos de la temporada más reciente agrupados por división, por orden alfabético."""
        grouped: dict[str, list[str]] = {division: [] for division in DIVISIONS}
        for team, division in sorted(self.current_division.items()):
            grouped.setdefault(division, []).append(team)
        return grouped

    def _division_of(self, team: str) -> str:
        if team in self.current_division:
            return self.current_division[team]
        return str(self._long[self._long["team"] == team]["division"].iloc[-1])

    def overview(self, team: str) -> dict:
        """Ficha de un equipo: Elo, balance, forma, últimos partidos y recorrido por temporadas."""
        team = resolve_team(self.raw, team)
        matches = self._long[self._long["team"] == team]
        recent = matches.tail(5).iloc[::-1]
        return {
            "team": team,
            "division": self._division_of(team),
            "active": team in self.current_division,
            "elo": round(self.ratings[team], 1),
            "elo_rank": self._rank.get(team),
            "elo_rank_of": len(self._rank),
            "latest_season": self.latest_season,
            "current_season": team_summary(self.raw, team, season=self.latest_season)
            if (matches["season"] == self.latest_season).any()
            else None,
            "all_time": team_summary(self.raw, team),
            "form": form_string(self.raw, team, 5),
            "last_matches": [
                {
                    "date": row.date.date().isoformat(),
                    "opponent": row.opponent,
                    "venue": row.venue,
                    "score": f"{row.goals_for}-{row.goals_against}",
                    "outcome": row.outcome,
                }
                for row in recent.itertuples()
            ],
            "seasons": team_by_season(self.raw, team).to_dict("records"),
        }

    # --- predicción ----------------------------------------------------------------------
    def predict(self, home: str, away: str, division: str | None = None) -> dict:
        """Probabilidades 1X2 de un partido entre dos equipos de los datos.

        ``division`` condiciona el modelo (en Segunda hay más empates y menos ventaja local);
        por defecto se usa la división actual del equipo local. Si los equipos no juegan en la
        misma división el partido es hipotético y se avisa.
        """
        home = resolve_team(self.raw, home)
        away = resolve_team(self.raw, away)
        if home == away:
            raise ValueError("El equipo local y el visitante deben ser distintos")
        if division is not None and division not in DIVISIONS:
            raise ValueError(f"División no válida: {division!r} (usa SP1 o SP2)")
        home_division, away_division = self._division_of(home), self._division_of(away)
        used_division = division or home_division

        row = pd.DataFrame(
            {
                "elo_diff": [self.ratings[home] - self.ratings[away]],
                "division": [used_division],
            }
        )
        probs = self.model.predict_proba(row).iloc[0]
        chances = {"home": float(probs["p_home"]), "draw": float(probs["p_draw"])}
        chances["away"] = float(probs["p_away"])
        return {
            "home": home,
            "away": away,
            "division": used_division,
            "home_elo": round(self.ratings[home], 1),
            "away_elo": round(self.ratings[away], 1),
            "probabilities": {key: round(value, 4) for key, value in chances.items()},
            "fair_odds": {key: round(1.0 / value, 2) for key, value in chances.items()},
            "hypothetical": home_division != away_division,
        }


__all__ = ["DIVISIONS", "PredictorService", "UnknownTeamError"]
