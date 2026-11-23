"""Servicio que junta datos, Elo y modelo, y responde lo que necesita la web.

Se construye una vez al arrancar (``PredictorService(load_all(...))``) y después responde sin
volver a leer ni a entrenar. Devuelve diccionarios y listas simples, listos para plantillas o
JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchpredictor.analysis import (
    UnknownTeamError,
    form_string,
    resolve_team,
    team_by_season,
    team_matches,
    team_summary,
)
from matchpredictor.features import add_form_features, compute_elo, current_form
from matchpredictor.models import FeatureLogit

DIVISIONS = ("SP1", "SP2")
#: features del modelo: diferencia de Elo y de tiros a puerta recientes (ver evaluation.backtest)
MODEL_FEATURES = ["elo_diff", "form_sot_diff"]


class PredictorService:
    def __init__(self, raw: pd.DataFrame, division_offset: float = 0.0):
        """``division_offset``: un rating de Segunda equivale a (rating + offset) en La Liga.

        Se aplica al calcular el Elo en los ascensos y descensos y, sobre todo, al comparar
        equipos de divisiones distintas (ver ``compute_elo`` y ``evaluation.divisions``).
        """
        if raw.empty:
            raise ValueError("No hay partidos con los que trabajar")
        self.raw = raw
        self.division_offset = float(division_offset)
        self.games, self.ratings = compute_elo(raw, division_offset=self.division_offset)
        self.games = add_form_features(self.games)
        self.model = FeatureLogit(MODEL_FEATURES).fit(self.games)
        #: forma actual por equipo (para predecir partidos futuros)
        self.form = current_form(self.games)
        self.latest_season = str(raw["season"].max())
        self._long = team_matches(raw)
        in_season = self._long[self._long["season"] == self.latest_season]
        #: división actual de cada equipo que juega en la temporada más reciente
        self.current_division = in_season.groupby("team")["division"].last().to_dict()
        self._rank = {
            team: position
            for position, team in enumerate(
                sorted(self.current_division, key=lambda t: -self._common_rating(t)), start=1
            )
        }

    def _common_rating(self, team: str) -> float:
        """Rating en la escala de La Liga, para comparar equipos de distintas divisiones."""
        offset = self.division_offset if self._division_of(team) == "SP2" else 0.0
        return self.ratings[team] + offset

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

    def _sot_form(self, team: str) -> float | None:
        """Diferencia media de tiros a puerta por partido en los últimos 10 (None sin datos)."""
        value = self.form.loc[team, "form_sot"]
        return None if pd.isna(value) else round(float(value), 1)

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
            "sot_form": self._sot_form(team),
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

        hypothetical = home_division != away_division
        if hypothetical:  # equipos de distinta división: ambos en la escala de La Liga
            elo_diff = self._common_rating(home) - self._common_rating(away)
        else:
            elo_diff = self.ratings[home] - self.ratings[away]
        sot_home, sot_away = self._sot_form(home), self._sot_form(away)
        sot_diff = np.nan if sot_home is None or sot_away is None else sot_home - sot_away
        row = pd.DataFrame(
            {"elo_diff": [elo_diff], "form_sot_diff": [sot_diff], "division": [used_division]}
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
            "sot_form": {"home": sot_home, "away": sot_away},
            "hypothetical": hypothetical,
            "division_offset": self.division_offset if hypothetical else 0.0,
        }


__all__ = ["DIVISIONS", "PredictorService", "UnknownTeamError"]
