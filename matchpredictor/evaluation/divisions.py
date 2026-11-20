"""¿Están en la misma escala los ratings de La Liga y de Segunda?

El Elo solo conecta las dos divisiones a través de los equipos que suben y bajan. Aquí se mide
si esa conexión es razonable: un equipo recién ascendido llega a La Liga con el rating que
ganó en Segunda; si luego rinde peor de lo que ese rating predice, los ratings de Segunda están
"inflados" respecto a La Liga (y al revés para los descendidos).

Se estima un desfase ``d``: *un rating de Segunda equivale a (rating + d) en La Liga*. Un ``d``
negativo significa que Segunda está inflada. Solo se miran los primeros partidos de cada
equipo en su nueva división, antes de que el Elo se adapte.

    python -m matchpredictor.evaluation.divisions --data data/raw
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd

from matchpredictor.analysis import team_matches
from matchpredictor.features.elo import expected_score

HOME_ADVANTAGE = 60.0  # el mismo valor por defecto de compute_elo


def transitions(games: pd.DataFrame) -> pd.DataFrame:
    """Ascensos (SP2 -> SP1) y descensos (SP1 -> SP2) entre temporadas consecutivas.

    Solo cuentan los equipos que aparecen en los datos la temporada anterior: uno que viene
    de Segunda B no es un ascenso entre las dos divisiones.
    """
    seasons = sorted(games["season"].unique())
    previous = dict(zip(seasons[1:], seasons[:-1], strict=True))
    per_season = team_matches(games).groupby(["team", "season"])["division"].first()
    rows = []
    for (team, season), division in per_season.items():
        before = previous.get(season)
        if before is None or (team, before) not in per_season.index:
            continue
        origin = per_season[(team, before)]
        if origin == "SP2" and division == "SP1":
            rows.append((team, season, "promoted"))
        elif origin == "SP1" and division == "SP2":
            rows.append((team, season, "relegated"))
    return pd.DataFrame(rows, columns=["team", "season", "kind"])


def first_matches(games: pd.DataFrame, k: int = 10, home_advantage: float = HOME_ADVANTAGE):
    """Primeros ``k`` partidos de cada equipo ascendido o descendido en su nueva división.

    Devuelve una fila por partido con ``diff`` (rating del equipo menos el del rival, con
    ventaja de campo, previos al partido), ``actual`` (1 victoria, 0,5 empate, 0 derrota) y
    ``sign`` (+1 ascendido, -1 descendido).
    """
    if "elo_diff" not in games.columns:
        raise ValueError("games debe traer las columnas de Elo (usa compute_elo)")
    moved = transitions(games)
    score = {"H": 1.0, "D": 0.5, "A": 0.0}
    home = pd.DataFrame(
        {
            "team": games["home_team"],
            "season": games["season"],
            "date": games["date"],
            "diff": games["elo_diff"] + home_advantage,
            "actual": games["result"].map(score),
        }
    )
    away = pd.DataFrame(
        {
            "team": games["away_team"],
            "season": games["season"],
            "date": games["date"],
            "diff": -(games["elo_diff"] + home_advantage),
            "actual": 1.0 - games["result"].map(score),
        }
    )
    both = pd.concat([home, away], ignore_index=True).merge(moved, on=["team", "season"])
    both = both.sort_values("date", kind="stable")
    both = both.groupby(["team", "season"], sort=False).head(k)
    both["sign"] = np.where(both["kind"] == "promoted", 1.0, -1.0)
    return both.reset_index(drop=True)[["team", "season", "kind", "diff", "actual", "sign"]]


def estimate_offset(
    diff: Sequence[float],
    actual: Sequence[float],
    sign: Sequence[float],
    low: float = -800.0,
    high: float = 800.0,
) -> float:
    """Desfase ``d`` que hace que, en media, lo esperado coincida con lo ocurrido.

    Resuelve  Σ sign·(actual − E(diff + sign·d)) = 0  por bisección (la suma decrece con ``d``).
    Devuelve NaN si no hay datos o la solución cae fuera de [low, high].
    """
    diff, actual, sign = (np.asarray(x, dtype=float) for x in (diff, actual, sign))
    if diff.size == 0:
        return float("nan")

    def gap(d: float) -> float:
        return float(np.sum(sign * (actual - expected_score(diff + sign * d))))

    if gap(low) < 0 or gap(high) > 0:
        return float("nan")
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if gap(middle) > 0 else (low, middle)
    return (low + high) / 2


def bootstrap_interval(
    rows: pd.DataFrame, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Intervalo del desfase remuestreando equipos-temporada completos (no partidos sueltos)."""
    groups = [
        group[["diff", "actual", "sign"]].to_numpy()
        for _, group in rows.groupby(["team", "season"])
    ]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sample = np.vstack([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        value = estimate_offset(sample[:, 0], sample[:, 1], sample[:, 2])
        if not np.isnan(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    return float(np.percentile(estimates, tail)), float(np.percentile(estimates, 100 - tail))


def _describe(rows: pd.DataFrame) -> dict:
    expected = expected_score(rows["diff"].to_numpy())
    return {
        "teams": rows.groupby(["team", "season"]).ngroups,
        "matches": len(rows),
        "actual": rows["actual"].mean(),
        "expected": float(np.mean(expected)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    from matchpredictor.data import load_all
    from matchpredictor.features import compute_elo

    parser = argparse.ArgumentParser(description="Escala de Elo entre La Liga y Segunda")
    parser.add_argument("--data", default="data/raw", help="directorio con los CSV")
    parser.add_argument("--first", type=int, default=10, help="primeros partidos a mirar")
    args = parser.parse_args(argv)

    games, _ = compute_elo(load_all(args.data))
    rows = first_matches(games, k=args.first)
    if rows.empty:
        print("No hay ascensos ni descensos entre temporadas consecutivas en los datos.")
        return 0

    print(f"Primeros {args.first} partidos de cada equipo tras cambiar de división\n")
    for kind, label in (
        ("promoted", "Ascendidos (Segunda -> La Liga)"),
        ("relegated", "Descendidos (La Liga -> Segunda)"),
    ):
        part = rows[rows["kind"] == kind]
        if part.empty:
            print(f"{label}: sin casos")
            continue
        info = _describe(part)
        own = estimate_offset(part["diff"], part["actual"], part["sign"])
        print(
            f"{label}: {info['teams']} equipos, {info['matches']} partidos | "
            f"puntos reales {info['actual']:.3f} vs esperados {info['expected']:.3f} | "
            f"desfase d implícito: {own:+.0f}"
        )

    estimate = estimate_offset(rows["diff"], rows["actual"], rows["sign"])
    low, high = bootstrap_interval(rows)
    print(f"\nDesfase combinado d = {estimate:+.0f} puntos Elo (IC 95 %: {low:+.0f} a {high:+.0f})")
    print("d < 0: los ratings de Segunda están inflados respecto a La Liga; d > 0: desinflados.")
    if low <= 0 <= high:
        print("El intervalo incluye 0: no hay evidencia de que haga falta corregir la escala.")
    print(
        "Ojo: el Elo se adapta durante esos partidos, así que esto tiende a subestimar |d|; "
        "con pocos equipos el margen de error es amplio."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
