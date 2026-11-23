"""Ajuste de los parámetros del Elo sin hacer trampa.

Los parámetros se eligen con unas temporadas de **validación** y se comprueban en otras de
**holdout** que no se han usado para elegir nada. Así la mejora que se reporta no es fruto de
haber probado muchas combinaciones sobre los mismos partidos.

    python -m matchpredictor.evaluation.tuning --data data/raw
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Mapping, Sequence

import pandas as pd

from matchpredictor.evaluation.backtest import complete_seasons, run_backtest, score_table
from matchpredictor.evaluation.metrics import PROB_COLUMNS, log_loss
from matchpredictor.features import compute_elo
from matchpredictor.models import EloLogit, FrequencyBaseline

DEFAULT_PARAMS = {
    "k": 20.0,
    "home_advantage": 60.0,
    "season_regression": 0.0,
    "sp2_initial": 1400.0,
    "division_offset": 0.0,
}
DEFAULT_GRID = {
    "k": [10, 15, 20, 25],
    "home_advantage": [60],  # la ventaja local la absorbe la regresión logística: no influye
    "season_regression": [0.0, 0.15, 0.3],
    "sp2_initial": [1200, 1300, 1400],
    "division_offset": [0, -15, -30, -45, -60],
}
QUICK_GRID = {
    "k": [10, 20],
    "home_advantage": [60],
    "season_regression": [0.0],
    "sp2_initial": [1400],
    "division_offset": [0],
}


def elo_games(raw: pd.DataFrame, params: Mapping[str, float]) -> pd.DataFrame:
    """Calcula el Elo con unos parámetros (``sp2_initial`` es el rating inicial en Segunda)."""
    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"Parámetros desconocidos: {sorted(unknown)}")
    values = {**DEFAULT_PARAMS, **params}
    games, _ = compute_elo(
        raw,
        k=values["k"],
        home_advantage=values["home_advantage"],
        season_regression=values["season_regression"],
        division_offset=values["division_offset"],
        initial={"SP1": 1500.0, "SP2": values["sp2_initial"]},
    )
    return games


def score_params(raw: pd.DataFrame, params: Mapping[str, float], seasons: Sequence[str]) -> float:
    """Log-loss de ``EloLogit`` con esos parámetros en ``seasons`` (entrenando con las previas)."""
    predictions = run_backtest(
        elo_games(raw, params), {"elo": EloLogit}, seasons, include_market=False
    )
    return log_loss(predictions[PROB_COLUMNS], predictions["result"])


def grid_search(
    raw: pd.DataFrame, grid: Mapping[str, Sequence[float]], seasons: Sequence[str]
) -> pd.DataFrame:
    """Prueba todas las combinaciones de ``grid``; devuelve una fila por combinación, de mejor
    (menor log-loss) a peor."""
    names = list(grid)
    rows = []
    for combo in itertools.product(*(grid[name] for name in names)):
        params = dict(zip(names, combo, strict=True))
        rows.append({**params, "log_loss": score_params(raw, params, seasons)})
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").reset_index(drop=True)


def edge_parameters(best: Mapping[str, float], grid: Mapping[str, Sequence[float]]) -> list[str]:
    """Parámetros cuyo mejor valor está en el borde de la rejilla (conviene ampliarla)."""
    return [
        name
        for name, values in grid.items()
        if len(values) > 1 and best[name] in (min(values), max(values))
    ]


def main(argv: Sequence[str] | None = None) -> int:
    from matchpredictor.data import load_all

    parser = argparse.ArgumentParser(description="Ajuste del Elo con validación y holdout")
    parser.add_argument("--data", default="data/raw", help="directorio con los CSV")
    parser.add_argument("--holdout", type=int, default=4, help="temporadas reservadas al final")
    parser.add_argument("--quick", action="store_true", help="rejilla mínima (para probar)")
    args = parser.parse_args(argv)

    raw = load_all(args.data)
    seasons = complete_seasons(raw)
    holdout = seasons[-args.holdout :]
    validation = seasons[2 : -args.holdout]  # las dos primeras solo aportan historia
    if not validation:
        raise SystemExit("No quedan temporadas de validación: reduce --holdout")
    grid = QUICK_GRID if args.quick else DEFAULT_GRID

    print(f"Validación: {', '.join(validation)} | Holdout: {', '.join(holdout)}")
    print(f"Probando {len(list(itertools.product(*grid.values())))} combinaciones...")
    results = grid_search(raw, grid, validation)
    best = {name: results.iloc[0][name] for name in grid}

    pd.set_option("display.width", 120)
    print("\n== MEJORES COMBINACIONES (log-loss en validación) ==")
    print(results.head(10).round(4).to_string(index=False))
    default_score = score_params(raw, DEFAULT_PARAMS, validation)
    print(f"\nParámetros actuales {DEFAULT_PARAMS}: validación {default_score:.4f}")
    edges = edge_parameters(best, grid)
    if edges:
        print(f"Aviso: el mejor valor está en el borde de la rejilla para: {', '.join(edges)}")

    default_run = run_backtest(
        elo_games(raw, DEFAULT_PARAMS),
        {"frecuencias": FrequencyBaseline, "elo_actual": EloLogit},
        holdout,
    )
    tuned_run = run_backtest(
        elo_games(raw, best), {"elo_ajustado": EloLogit}, holdout, include_market=False
    )
    predictions = pd.concat([default_run, tuned_run], ignore_index=True)
    for title, by in (
        ("GLOBAL", []),
        ("POR TEMPORADA", ["season"]),
        ("POR DIVISIÓN", ["division"]),
    ):
        print(f"\n== HOLDOUT {title} (partidos con cuotas; no se ha usado para elegir) ==")
        print(score_table(predictions, by).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
