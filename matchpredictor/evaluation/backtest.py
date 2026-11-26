"""Backtesting temporal: cada temporada de test se predice solo con datos de temporadas
anteriores, que es lo que pasaría de verdad al usar el modelo.

    python -m matchpredictor.evaluation.backtest --data data/raw
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from matchpredictor.evaluation.metrics import (
    PROB_COLUMNS,
    brier_score,
    log_loss,
    market_probabilities,
)

_MATCH_KEY = ["date", "home_team", "away_team"]
_PREDICTION_COLUMNS = ["season", "division", *_MATCH_KEY, "result", "model", *PROB_COLUMNS]


def complete_seasons(games: pd.DataFrame, min_share: float = 0.95) -> list[str]:
    """Temporadas completas: en cada división tienen casi tantos partidos como la mayor."""
    counts = games.groupby(["division", "season"]).size().unstack("division")
    full = counts.ge(counts.max() * min_share).all(axis=1)
    return sorted(counts.index[full])


def run_backtest(
    games: pd.DataFrame,
    models: Mapping[str, Callable[[], object]],
    test_seasons: Sequence[str],
    include_market: bool = True,
) -> pd.DataFrame:
    """Predicciones fuera de muestra de cada modelo (y del mercado) en ``test_seasons``.

    ``models`` asocia un nombre a una función sin argumentos que devuelve un modelo nuevo con
    ``fit`` y ``predict_proba``. Para cada temporada se entrena solo con las anteriores. Las
    temporadas sin historia previa se saltan. Devuelve un DataFrame largo (una fila por
    partido y modelo).
    """
    pieces = []
    for season in sorted(test_seasons):
        train = games[games["season"] < season]
        test = games[games["season"] == season]
        if train.empty or test.empty:
            continue
        for name, factory in models.items():
            probs = factory().fit(train).predict_proba(test)
            pieces.append(_with_meta(test, probs, name))
        if include_market:
            market = market_probabilities(test)
            has_odds = market["p_home"].notna()
            if has_odds.any():
                pieces.append(_with_meta(test[has_odds], market[has_odds], "market"))
    if not pieces:
        raise ValueError("No hay temporadas de test con historia previa para entrenar")
    return pd.concat(pieces, ignore_index=True)[_PREDICTION_COLUMNS]


def _with_meta(test: pd.DataFrame, probs: pd.DataFrame, name: str) -> pd.DataFrame:
    meta = test[["season", "division", *_MATCH_KEY, "result"]].copy()
    meta["model"] = name
    return pd.concat([meta, probs[PROB_COLUMNS]], axis=1)


def score_table(
    predictions: pd.DataFrame, by: Sequence[str] = (), common: bool = True
) -> pd.DataFrame:
    """Log-loss y Brier por modelo (y por los grupos de ``by``, p. ej. ``["season"]``).

    Con ``common=True`` solo cuentan los partidos que tienen predicción de **todos** los
    modelos (normalmente los que tienen cuotas), para que la comparación sea justa.
    """
    if common:
        n_models = predictions["model"].nunique()
        per_match = predictions.groupby(_MATCH_KEY)["model"].transform("nunique")
        predictions = predictions[per_match == n_models]
    rows = []
    for keys, group in predictions.groupby([*by, "model"], sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip([*by, "model"], keys, strict=True)),
                "matches": len(group),
                "log_loss": log_loss(group[PROB_COLUMNS], group["result"]),
                "brier": brier_score(group[PROB_COLUMNS], group["result"]),
            }
        )
    return pd.DataFrame(rows)


def track_backtest(predictions: pd.DataFrame, seasons: Sequence[str]) -> str:
    """Registra en MLflow el log-loss y el Brier de cada modelo (solo partidos con cuotas)."""
    from matchpredictor.tracking import log_run

    table = score_table(predictions)
    metrics = {}
    for row in table.itertuples():
        metrics[f"{row.model}.log_loss"] = row.log_loss
        metrics[f"{row.model}.brier"] = row.brier
    return log_run(
        run_name=f"backtest {seasons[0]}..{seasons[-1]}",
        params={"seasons": ",".join(seasons), "models": ",".join(sorted(table["model"]))},
        metrics={**metrics, "matches": float(table["matches"].max())},
        tags={"kind": "backtest"},
    )


def main(argv: Sequence[str] | None = None) -> int:
    from matchpredictor.data import load_all
    from matchpredictor.features import add_form_features, compute_elo
    from matchpredictor.models import EloLogit, FeatureLogit, FrequencyBaseline

    parser = argparse.ArgumentParser(description="Backtesting temporal de la línea base")
    parser.add_argument("--data", default="data/raw", help="directorio con los CSV")
    parser.add_argument("--last", type=int, default=5, help="número de temporadas de test")
    parser.add_argument(
        "--track", action="store_true", help="registra el resultado en MLflow (mlruns/)"
    )
    args = parser.parse_args(argv)

    games, _ = compute_elo(load_all(args.data))
    games = add_form_features(games)
    seasons = complete_seasons(games)[-args.last :]
    predictions = run_backtest(
        games,
        {
            "frecuencias": FrequencyBaseline,
            "elo_logit": EloLogit,
            "elo_sot": lambda: FeatureLogit(["elo_diff", "form_sot_diff"]),
        },
        test_seasons=seasons,
    )
    pd.set_option("display.width", 120)
    print(f"Temporadas de test: {', '.join(seasons)} (entrenando siempre solo con las anteriores)")
    for title, by in (
        ("GLOBAL", []),
        ("POR TEMPORADA", ["season"]),
        ("POR DIVISIÓN", ["division"]),
    ):
        print(f"\n== {title} (solo partidos con cuotas) ==")
        print(score_table(predictions, by).round(4).to_string(index=False))
    if args.track:
        run_id = track_backtest(predictions, seasons)
        print(f"\nRegistrado en MLflow (run {run_id}). Míralo con:")
        print("  mlflow ui --backend-store-uri file:mlruns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
