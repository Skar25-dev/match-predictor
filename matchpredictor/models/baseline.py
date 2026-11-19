"""Modelos de línea base. Todos exponen ``fit(games)`` y ``predict_proba(games)``.

``games`` es el DataFrame de ``compute_elo`` (con ``home_elo``, ``away_elo``, ``elo_diff``).
``predict_proba`` devuelve un DataFrame ``p_home``, ``p_draw``, ``p_away`` con el índice de
``games``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from matchpredictor.evaluation.metrics import OUTCOMES, PROB_COLUMNS


def _to_frame(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=PROB_COLUMNS)


class FrequencyBaseline:
    """Predice siempre las frecuencias históricas de local / empate / visitante."""

    def fit(self, games: pd.DataFrame) -> FrequencyBaseline:
        shares = games["result"].value_counts(normalize=True)
        self.probs_ = np.array([shares.get(outcome, 0.0) for outcome in OUTCOMES])
        return self

    def predict_proba(self, games: pd.DataFrame) -> pd.DataFrame:
        return _to_frame(np.tile(self.probs_, (len(games), 1)), games.index)


class EloLogit:
    """Regresión logística multinomial sobre la diferencia de Elo (y la división).

    La división entra como variable porque en Segunda hay más empates y menos ventaja de
    campo que en La Liga, y el Elo por sí solo no lo capta.
    """

    def __init__(self, c: float = 1.0):
        self.c = c

    @staticmethod
    def _features(games: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                games["elo_diff"].to_numpy(dtype=float) / 100.0,
                (games["division"] == "SP2").to_numpy(),
            ]
        ).astype(float)

    def fit(self, games: pd.DataFrame) -> EloLogit:
        self.model_ = LogisticRegression(C=self.c, max_iter=1000)
        self.model_.fit(self._features(games), games["result"])
        return self

    def predict_proba(self, games: pd.DataFrame) -> pd.DataFrame:
        raw = self.model_.predict_proba(self._features(games))
        order = [list(self.model_.classes_).index(outcome) for outcome in OUTCOMES]
        return _to_frame(raw[:, order], games.index)
