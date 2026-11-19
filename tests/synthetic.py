"""Liga sintética con fuerzas conocidas, para probar modelos y backtesting sin datos reales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchpredictor.features import compute_elo
from matchpredictor.features.elo import expected_score

DRAW_PROB = 0.26
MARGIN = 1.06


def synthetic_raw(n_seasons: int = 5, n_teams: int = 14, seed: int = 0) -> pd.DataFrame:
    """Partidos de ``n_seasons`` temporadas de una división SP1, con cuotas B365 (con margen).

    Son datos \"crudos\": como los de ``load_all``, sin columnas de Elo."""
    rng = np.random.default_rng(seed)
    strengths = rng.normal(0, 200, n_teams)
    rows = []
    for s in range(n_seasons):
        year = 2019 + s
        day = 0
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                e = expected_score(strengths[i] + 60 - strengths[j])
                probs = np.array([(1 - DRAW_PROB) * e, DRAW_PROB, (1 - DRAW_PROB) * (1 - e)])
                result = rng.choice(["H", "D", "A"], p=probs)
                goals = {"H": (2, 1), "D": (1, 1), "A": (0, 1)}[result]
                odds = 1.0 / (probs * MARGIN)
                rows.append(
                    {
                        "season": f"{year}-{(year + 1) % 100:02d}",
                        "division": "SP1",
                        "date": pd.Timestamp(f"{year}-08-15") + pd.Timedelta(days=day // 7),
                        "time": pd.NA,
                        "home_team": f"T{i:02d}",
                        "away_team": f"T{j:02d}",
                        "home_goals": goals[0],
                        "away_goals": goals[1],
                        "result": result,
                        "b365_home": odds[0],
                        "b365_draw": odds[1],
                        "b365_away": odds[2],
                    }
                )
                day += 1
    return pd.DataFrame(rows)


def synthetic_games(n_seasons: int = 5, n_teams: int = 14, seed: int = 0) -> pd.DataFrame:
    """Lo mismo que ``synthetic_raw`` pero con el Elo calculado."""
    games, _ = compute_elo(synthetic_raw(n_seasons, n_teams, seed))
    return games


def write_as_csvs(games: pd.DataFrame, directory) -> None:
    """Escribe ``games`` en el formato de football-data.co.uk, un CSV por temporada."""
    for season, group in games.groupby("season"):
        frame = pd.DataFrame(
            {
                "Div": group["division"],
                "Date": group["date"].dt.strftime("%d/%m/%Y"),
                "HomeTeam": group["home_team"],
                "AwayTeam": group["away_team"],
                "FTHG": group["home_goals"],
                "FTAG": group["away_goals"],
                "FTR": group["result"],
                "B365H": group["b365_home"],
                "B365D": group["b365_draw"],
                "B365A": group["b365_away"],
            }
        )
        frame.to_csv(directory / f"SP1_{season}.csv", index=False)
