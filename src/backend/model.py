from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


DEFAULT_FEATURES = ["voltage", "current", "frequency"]


@dataclass(frozen=True)
class ModelConfig:
    feature_names: Tuple[str, ...] = tuple(DEFAULT_FEATURES)
    random_state: int = 42
    contamination: float = 0.05
    n_estimators: int = 200


class AnomalyModel:
    def __init__(self, model: IsolationForest, config: ModelConfig) -> None:
        self.model = model
        self.config = config

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self.config.feature_names

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (anomaly_flags, anomaly_scores)
        - anomaly_flags: 1 for anomaly, 0 for normal
        - anomaly_scores: higher means more anomalous (derived from -decision_function)
        """
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected 2D array with {len(self.feature_names)} features, got {X.shape}"
            )

        preds = self.model.predict(X)
        scores = -self.model.decision_function(X)
        flags = (preds == -1).astype(int)
        return flags, scores


def _generate_synthetic_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic NORMAL operating points for training."""
    voltage = rng.normal(230.0, 3.5, size=n)
    current = rng.normal(10.0, 2.0, size=n)
    frequency = rng.normal(50.0, 0.05, size=n)
    return np.vstack([voltage, current, frequency]).T


def train_isolation_forest(
    config: ModelConfig, n_samples: int = 5000
) -> IsolationForest:
    rng = np.random.default_rng(config.random_state)
    X = _generate_synthetic_normal(n_samples, rng)
    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
    )
    model.fit(X)
    return model


def ensure_model(
    model_path: Path | str,
    config: ModelConfig | None = None,
    train_if_missing: bool = True,
) -> AnomalyModel:
    path = Path(model_path)
    cfg = config or ModelConfig()
    if path.exists():
        model = joblib.load(path)
        return AnomalyModel(model, cfg)
    if not train_if_missing:
        raise FileNotFoundError(f"Model artifact not found at {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    model = train_isolation_forest(cfg)
    joblib.dump(model, path)
    return AnomalyModel(model, cfg)


def as_feature_matrix(
    items: Iterable[dict], feature_order: Iterable[str]
) -> np.ndarray:
    order = list(feature_order)
    mat = np.array([[float(item[f]) for f in order] for item in items], dtype=float)
    return mat


def default_model_path() -> Path:
    base = Path(os.getenv("MODEL_DIR", "models"))
    name = os.getenv("MODEL_NAME", "iso_forest.joblib")
    return base / name
