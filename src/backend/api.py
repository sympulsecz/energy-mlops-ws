from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    voltage: float = Field(..., description="Voltage in Volts")
    current: float = Field(..., description="Current in Amperes")
    frequency: float = Field(..., description="Grid frequency in Hz")
    timestamp: Optional[str] = Field(None, description="ISO timestamp (optional)")


class BatchPredictRequest(BaseModel):
    readings: List[SensorReading]


class Prediction(BaseModel):
    anomaly: bool
    score: float


class BatchPredictResponse(BaseModel):
    predictions: List[Prediction]
    anomalies: int
    total: int
    served_by: str | None = None
