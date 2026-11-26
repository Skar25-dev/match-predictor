"""Registro de predicciones: guarda lo que el modelo dice *antes* de cada partido y lo compara
después con el resultado y con el mercado.

    python -m matchpredictor.evaluation.predictions_log record      # guarda el calendario
    python -m matchpredictor.evaluation.predictions_log report      # compara con resultados
    python -m matchpredictor.evaluation.predictions_log replay --since 2026-08-01

``record`` lee ``data/fixtures.csv`` y añade al registro las predicciones de los partidos aún
no jugados. Un partido ya registrado no se vuelve a escribir: se conserva la primera predicción,
la que se hizo antes del partido y no se puede retocar con información posterior.

``replay`` simula el registro sobre partidos ya jugados: para cada semana entrena solo con los
partidos anteriores y predice los de esa semana (sin mirar el futuro). Sirve para tener datos
reales ya, y esas filas quedan marcadas con ``mode = replay``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from matchpredictor.evaluation.metrics import PROB_COLUMNS, brier_score, log_loss

DEFAULT_LOG_PATH = Path("data/predictions.csv")
LOG_COLUMNS = [
    "made_on",
    "mode",
    "division",
    "date",
    "home_team",
    "away_team",
    "model",
    "p_home",
    "p_draw",
    "p_away",
    "market_source",
    "m_home",
    "m_draw",
    "m_away",
]
_KEY = ["date", "home_team", "away_team"]
# Los partidos futuros solo tienen cuotas de apertura: en la repetición se descartan las de cierre.
_CLOSING_ODDS = ("b365c", "avgc", "psc")


def entries_to_log(entries: list[dict], made_on: str, mode: str = "live") -> pd.DataFrame:
    """Filas de registro a partir de ``PredictorService.predict_fixtures`` (sin los errores)."""
    rows = []
    for entry in entries:
        if entry["error"] is not None:
            continue
        probs, market = entry["probabilities"], entry["market"] or {}
        rows.append(
            {
                "made_on": made_on,
                "mode": mode,
                "division": entry["division"],
                "date": entry["date"],
                "home_team": entry["home"],
                "away_team": entry["away"],
                "model": entry["model"],
                "p_home": probs["home"],
                "p_draw": probs["draw"],
                "p_away": probs["away"],
                "market_source": entry["market_source"],
                "m_home": market.get("home"),
                "m_draw": market.get("draw"),
                "m_away": market.get("away"),
            }
        )
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def read_log(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, dtype={"made_on": "string", "date": "string"})


def append_to_log(path: str | Path, new: pd.DataFrame) -> int:
    """Añade al registro los partidos que aún no estaban y devuelve cuántos se añadieron.

    Si un partido ya está registrado se conserva la predicción original.
    """
    path = Path(path)
    current = read_log(path)
    known = set(zip(current["date"], current["home_team"], current["away_team"], strict=True))
    fresh = new[
        [
            key not in known
            for key in zip(new["date"], new["home_team"], new["away_team"], strict=True)
        ]
    ]
    fresh = fresh.drop_duplicates(_KEY)
    if not fresh.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([current, fresh], ignore_index=True)[LOG_COLUMNS].to_csv(path, index=False)
    return len(fresh)


def settle(log: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Registro con el resultado real (``result``) de los partidos que ya se han jugado."""
    played = raw[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]].copy()
    played["date"] = played["date"].dt.date.astype(str)
    merged = log.merge(played, on=_KEY, how="inner")
    return merged.reset_index(drop=True)


def score(settled: pd.DataFrame) -> pd.DataFrame:
    """Log-loss y Brier del modelo y del mercado en los mismos partidos (los que tienen cuotas)."""
    comparable = settled.dropna(subset=["m_home", "m_draw", "m_away"])
    rows = []
    for mode, group in [("todos", comparable), *comparable.groupby("mode")]:
        if group.empty:
            continue
        model = group[PROB_COLUMNS]
        market = group[["m_home", "m_draw", "m_away"]].set_axis(PROB_COLUMNS, axis=1)
        for name, probs in (("modelo", model), ("mercado", market)):
            rows.append(
                {
                    "registro": mode,
                    "predictor": name,
                    "partidos": len(group),
                    "log_loss": log_loss(probs, group["result"]),
                    "brier": brier_score(probs, group["result"]),
                }
            )
    return pd.DataFrame(rows)


def replay(raw: pd.DataFrame, since: str, step_days: int = 7) -> pd.DataFrame:
    """Predicciones de los partidos jugados desde ``since`` sin usar información posterior.

    Cada bloque de ``step_days`` días se predice con un modelo entrenado solo con los partidos
    anteriores al bloque.
    """
    from matchpredictor.data.loader import FIXTURE_COLUMNS
    from matchpredictor.service import PredictorService

    start = pd.Timestamp(since)
    target = raw[raw["date"] >= start]
    pieces = []
    block_start = start
    while block_start <= raw["date"].max():
        block_end = block_start + pd.Timedelta(days=step_days)
        block = target[(target["date"] >= block_start) & (target["date"] < block_end)]
        if not block.empty:
            history = raw[raw["date"] < block_start]
            fixtures = block.reindex(
                columns=[c for c in FIXTURE_COLUMNS if not c.startswith(_CLOSING_ODDS)]
            )
            entries = PredictorService(history).predict_fixtures(fixtures)
            pieces.append(
                entries_to_log(entries, made_on=block_start.date().isoformat(), mode="replay")
            )
        block_start = block_end
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=LOG_COLUMNS)


def main(argv: Sequence[str] | None = None) -> int:
    from matchpredictor.data import load_all, load_fixtures
    from matchpredictor.service import PredictorService

    parser = argparse.ArgumentParser(description="Registro de predicciones")
    parser.add_argument("command", choices=["record", "report", "replay"])
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--fixtures", default="data/fixtures.csv")
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--since", default=None, help="replay: fecha de inicio (AAAA-MM-DD)")
    parser.add_argument("--step-days", type=int, default=7, help="replay: días por bloque")
    args = parser.parse_args(argv)

    raw = load_all(args.data)
    if args.command == "record":
        if not Path(args.fixtures).exists():
            print(f"No existe {args.fixtures}. Descárgalo con: ... download --fixtures")
            return 1
        entries = PredictorService(raw).predict_fixtures(load_fixtures(args.fixtures))
        today = pd.Timestamp.today().date().isoformat()
        added = append_to_log(args.log, entries_to_log(entries, made_on=today))
        print(f"{added} predicciones nuevas guardadas en {args.log} ({len(entries)} en calendario)")
    elif args.command == "replay":
        if not args.since:
            print("Indica desde cuándo con --since AAAA-MM-DD")
            return 1
        added = append_to_log(args.log, replay(raw, args.since, args.step_days))
        print(f"{added} predicciones de repetición guardadas en {args.log}")
    else:
        settled = settle(read_log(args.log), raw)
        if settled.empty:
            print("Aún no hay partidos del registro con resultado.")
            return 0
        pd.set_option("display.width", 120)
        print(score(settled).round(4).to_string(index=False))
        pending = len(read_log(args.log)) - len(settled)
        print(f"\nPartidos con resultado: {len(settled)} | pendientes de jugarse: {pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
