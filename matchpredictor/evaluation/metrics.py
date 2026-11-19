"""Métricas de probabilidades 1X2 y probabilidades implícitas de las cuotas.

Las probabilidades se representan como un DataFrame con columnas ``p_home``, ``p_draw`` y
``p_away`` (en ese orden, que es también el de ``H``, ``D``, ``A`` en ``result``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
PROB_COLUMNS = ["p_home", "p_draw", "p_away"]

# Orden de preferencia de la casa de referencia: media de cierre, Bet365 de cierre, media de
# apertura, Bet365 de apertura. Pinnacle no se usa porque falta en buena parte de los datos.
MARKET_SOURCES = ("avgc", "b365c", "avg", "b365")


def _one_hot(results: pd.Series) -> np.ndarray:
    unknown = set(results) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"Resultados no válidos: {sorted(unknown)} (se esperan H, D, A)")
    return np.column_stack([(results == outcome).to_numpy() for outcome in OUTCOMES]).astype(float)


def _check(probs: pd.DataFrame, results: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if len(probs) != len(results):
        raise ValueError("probs y results deben tener la misma longitud")
    values = probs[PROB_COLUMNS].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Hay probabilidades vacías (NaN); filtra esos partidos antes")
    return values, _one_hot(results.reset_index(drop=True))


def log_loss(probs: pd.DataFrame, results: pd.Series, eps: float = 1e-15) -> float:
    """Log-loss medio (menor es mejor). Predecir 1/3 siempre da ln(3) ≈ 1,0986."""
    values, truth = _check(probs, results)
    chosen = np.clip((values * truth).sum(axis=1), eps, 1.0)
    return float(-np.log(chosen).mean())


def brier_score(probs: pd.DataFrame, results: pd.Series) -> float:
    """Brier multiclase medio (menor es mejor). Predecir 1/3 siempre da 2/3 ≈ 0,667."""
    values, truth = _check(probs, results)
    return float(((values - truth) ** 2).sum(axis=1).mean())


def implied_probabilities(home: pd.Series, draw: pd.Series, away: pd.Series) -> pd.DataFrame:
    """Probabilidades implícitas de unas cuotas decimales, quitando el margen de la casa.

    Se normaliza 1/cuota para que sumen 1. Si falta alguna cuota o es menor o igual que 1,
    esa fila queda en NaN.
    """
    odds = pd.concat([home, draw, away], axis=1, keys=PROB_COLUMNS).astype(float)
    valid = (odds > 1.0).all(axis=1)
    inverse = (1.0 / odds).where(valid)
    return inverse.div(inverse.sum(axis=1), axis=0)


def market_probabilities(
    df: pd.DataFrame, sources: tuple[str, ...] = MARKET_SOURCES
) -> pd.DataFrame:
    """Probabilidades del mercado por partido, usando la primera fuente disponible.

    Devuelve ``p_home``, ``p_draw``, ``p_away`` y ``market_source`` (qué cuotas se usaron).
    Los partidos sin ninguna cuota válida quedan en NaN.
    """
    result = pd.DataFrame(np.nan, index=df.index, columns=PROB_COLUMNS)
    result["market_source"] = pd.Series(pd.NA, index=df.index, dtype="object")
    for source in sources:
        columns = [f"{source}_home", f"{source}_draw", f"{source}_away"]
        if not all(column in df.columns for column in columns):
            continue
        probs = implied_probabilities(df[columns[0]], df[columns[1]], df[columns[2]])
        fill = result["p_home"].isna() & probs["p_home"].notna()
        result.loc[fill, PROB_COLUMNS] = probs.loc[fill, PROB_COLUMNS]
        result.loc[fill, "market_source"] = source
    return result
