from __future__ import annotations

import os
import time
from typing import List

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .api import BatchPredictRequest, BatchPredictResponse, Prediction
from .model import AnomalyModel, as_feature_matrix, default_model_path, ensure_model


logger = structlog.get_logger(__name__)


REQUEST_COUNT = Counter(
    "requests_total", "Total HTTP requests", ["path", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency (s)", ["path", "method"]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds", "Model inference latency (s)"
)
ANOMALY_COUNT = Counter("anomalies_total", "Total anomalies predicted")


def load_model() -> AnomalyModel:
    model_path = default_model_path()
    train_if_missing = os.getenv("TRAIN_IF_MISSING", "1") not in ("0", "false", "False")
    model = ensure_model(model_path, train_if_missing=train_if_missing)
    logger.info(
        "model_loaded", path=str(model_path), features=list(model.feature_names)
    )
    return model


model_instance: AnomalyModel | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Grid Anomaly Detection Service", version="0.1.0")

    @app.on_event("startup")
    def _startup() -> None:
        global model_instance
        model_instance = load_model()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": "isolation_forest",
            "features": list(model_instance.feature_names) if model_instance else [],
        }

    @app.post("/predict", response_model=BatchPredictResponse)
    def predict(req: BatchPredictRequest) -> BatchPredictResponse:
        if not req.readings:
            raise HTTPException(status_code=400, detail="No readings provided")

        if model_instance is None:
            raise HTTPException(status_code=503, detail="Model not ready")

        start = time.perf_counter()
        status_label = "200"
        try:
            X = as_feature_matrix(
                [r.model_dump() for r in req.readings], model_instance.feature_names
            )
            infer_t0 = time.perf_counter()
            flags, scores = model_instance.predict_batch(X)
            infer_dt = time.perf_counter() - infer_t0
            INFERENCE_LATENCY.observe(infer_dt)

            preds: List[Prediction] = [
                Prediction(anomaly=bool(flags[i]), score=float(scores[i]))
                for i in range(len(req.readings))
            ]
            anomalies = int(flags.sum())
            ANOMALY_COUNT.inc(anomalies)
            total = len(preds)
            return BatchPredictResponse(
                predictions=preds, anomalies=anomalies, total=total
            )
        except HTTPException as he:
            status_label = str(he.status_code)
            raise
        except Exception:
            status_label = "500"
            raise
        finally:
            req_dt = time.perf_counter() - start
            REQUEST_LATENCY.labels(path="/predict", method="POST").observe(req_dt)
            REQUEST_COUNT.labels(
                path="/predict", method="POST", status=status_label
            ).inc()

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        data = generate_latest()  # type: ignore[arg-type]
        return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(Exception)
    def handle_exc(_, exc: Exception):
        logger.exception("unhandled_exception", err=str(exc))
        REQUEST_COUNT.labels(path="*", method="*", status="500").inc()
        return JSONResponse(
            status_code=500, content={"detail": "Internal Server Error"}
        )

    return app


app = create_app()
