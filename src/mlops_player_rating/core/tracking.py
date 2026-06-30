"""Tiny MLflow bootstrap shared by the modelling pipelines.

Defaults to a local **SQLite** backend (``sqlite:///mlflow.db``) with a local artifact
folder (``./mlartifacts``). SQLite keeps the whole project reproducible with zero
infrastructure *and* unlocks the MLflow Model Registry (file stores are rejected by
MLflow 3.x and cannot register models). Point ``mlflow.tracking_uri`` at a remote
server in ``conf/local`` for a team setup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_mlflow(params: dict[str, Any]):
    """Configure the MLflow tracking URI + experiment and return the ``mlflow`` module."""
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = params.get("tracking_uri", "sqlite:///mlflow.db")

    # Resolve relative local backends against the cwd so the store is always found.
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db_abs = (Path.cwd() / uri[len("sqlite:///") :]).resolve()
        uri = f"sqlite:///{db_abs.as_posix()}"
    elif "://" not in uri:  # bare path -> file store
        path_like = Path(uri)
        if not path_like.is_absolute():
            path_like = Path.cwd() / path_like
        path_like.mkdir(parents=True, exist_ok=True)
        uri = path_like.as_uri()

    mlflow.set_tracking_uri(uri)

    exp_name = params.get("experiment_name", "player_rating")
    artifact_dir = (Path.cwd() / "mlartifacts").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    if client.get_experiment_by_name(exp_name) is None:
        client.create_experiment(exp_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(exp_name)

    logger.info("MLflow tracking_uri=%s experiment=%s", uri, exp_name)
    return mlflow
