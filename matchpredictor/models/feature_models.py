"""Modelos que usan varias features (Elo, forma, tiros, descanso).

Misma interfaz que los de ``baseline``: ``fit(games)`` y ``predict_proba(games)``. Los valores
vacíos de las features (primeros partidos, datos antiguos sin tiros) se tratan así:
la regresión logística los sustituye por la media del entrenamiento y el gradient boosting los
maneja de forma nativa.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from matchpredictor.evaluation.metrics import OUTCOMES, PROB_COLUMNS


def _matrix(games: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [name for name in features if name not in games.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features: {missing}")
    columns = [games[name].to_numpy(dtype=float) for name in features]
    columns.append((games["division"] == "SP2").to_numpy(dtype=float))
    return np.column_stack(columns)


def _ordered(probs: np.ndarray, classes: Sequence[str], index: pd.Index) -> pd.DataFrame:
    order = [list(classes).index(outcome) for outcome in OUTCOMES]
    return pd.DataFrame(probs[:, order], index=index, columns=PROB_COLUMNS)


class FeatureLogit:
    """Regresión logística multinomial sobre las features indicadas (y la división)."""

    def __init__(self, features: Sequence[str], c: float = 1.0):
        self.features = list(features)
        self.c = c

    def _scale(self, matrix: np.ndarray) -> np.ndarray:
        scaled = (matrix - self.mean_) / self.scale_
        return np.nan_to_num(scaled, nan=0.0)  # vacío -> media del entrenamiento

    def fit(self, games: pd.DataFrame) -> FeatureLogit:
        matrix = _matrix(games, self.features)
        with warnings.catch_warnings():  # una columna entera vacía (datos antiguos sin tiros)
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nan_to_num(np.nanmean(matrix, axis=0), nan=0.0)
            self.scale_ = np.nan_to_num(np.nanstd(matrix, axis=0), nan=1.0)
        self.scale_[self.scale_ == 0] = 1.0
        self.model_ = LogisticRegression(C=self.c, max_iter=1000)
        self.model_.fit(self._scale(matrix), games["result"])
        return self

    def predict_proba(self, games: pd.DataFrame) -> pd.DataFrame:
        raw = self.model_.predict_proba(self._scale(_matrix(games, self.features)))
        return _ordered(raw, self.model_.classes_, games.index)


class GradientBoosting:
    """Gradient boosting (``HistGradientBoostingClassifier``) con árboles poco profundos."""

    def __init__(self, features: Sequence[str], **params):
        self.features = list(features)
        self.params = {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 150,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
            "random_state": 0,
            **params,
        }

    def fit(self, games: pd.DataFrame) -> GradientBoosting:
        matrix = _matrix(games, self.features)
        # HistGradientBoosting falla con una columna entera vacía: se descarta (no aporta nada)
        self.keep_ = ~np.isnan(matrix).all(axis=0)
        self.model_ = HistGradientBoostingClassifier(**self.params)
        self.model_.fit(matrix[:, self.keep_], games["result"])
        return self

    def predict_proba(self, games: pd.DataFrame) -> pd.DataFrame:
        raw = self.model_.predict_proba(_matrix(games, self.features)[:, self.keep_])
        return _ordered(raw, self.model_.classes_, games.index)
