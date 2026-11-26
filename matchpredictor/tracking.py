"""Registro de experimentos con MLflow (opcional).

MLflow no es una dependencia obligatoria: solo se importa cuando se pide registrar algo
(``--track``). Instálalo con ``pip install -r requirements-tracking.txt`` y mira los resultados
con ``mlflow ui --backend-store-uri file:mlruns`` (carpeta ``mlruns/``, ya ignorada por git).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

DEFAULT_TRACKING_URI = "file:mlruns"
DEFAULT_EXPERIMENT = "match-predictor"


class TrackingUnavailableError(RuntimeError):
    """Se pidió registrar un experimento pero MLflow no está instalado."""


def git_commit() -> str | None:
    """Commit actual del repositorio (para saber con qué código se obtuvo un resultado)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def log_run(
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tags: Mapping[str, str] | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> str:
    """Guarda una ejecución en MLflow y devuelve su id. Las métricas NaN se omiten.

    Sin ``tracking_uri`` se usa la variable ``MLFLOW_TRACKING_URI`` y, si no existe, ``mlruns/``.
    """
    try:
        import mlflow
    except ImportError as error:
        raise TrackingUnavailableError(
            "Falta MLflow. Instálalo con: pip install -r requirements-tracking.txt"
        ) from error

    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )
    mlflow.set_experiment(experiment)
    all_tags = {"git_commit": git_commit() or "desconocido", **(tags or {})}
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        mlflow.set_tags(all_tags)
        return run.info.run_id
