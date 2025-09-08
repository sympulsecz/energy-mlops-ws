from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Generator, Optional


@dataclass
class SimConfig:
    voltage_base: float = 230.0
    voltage_noise: float = 3.5
    current_base: float = 10.0
    current_noise: float = 2.0
    frequency_base: float = 50.0
    frequency_noise: float = 0.05
    anomaly_rate: float = 0.02


def _normal(mu: float, sigma: float) -> float:
    return random.gauss(mu, sigma)


def _inject_anomaly(v: float, c: float, f: float) -> tuple[float, float, float]:
    t = random.choice(["v_spike", "v_dip", "c_spike", "c_dip", "f_spike", "f_dip"])
    if t == "v_spike":
        v += random.uniform(15, 40)
    elif t == "v_dip":
        v -= random.uniform(15, 40)
    elif t == "c_spike":
        c += random.uniform(8, 20)
    elif t == "c_dip":
        c -= random.uniform(8, 20)
    elif t == "f_spike":
        f += random.uniform(0.3, 0.8)
    elif t == "f_dip":
        f -= random.uniform(0.3, 0.8)
    return v, c, f


def stream_readings(
    cfg: Optional[SimConfig] = None,
) -> Generator[Dict[str, float], None, None]:
    cfg = cfg or SimConfig()
    t = 0
    while True:
        diurnal = math.sin(t / 200.0)
        v = _normal(cfg.voltage_base + 1.2 * diurnal, cfg.voltage_noise)
        c = _normal(cfg.current_base + 0.8 * diurnal, cfg.current_noise)
        f = _normal(cfg.frequency_base, cfg.frequency_noise)

        if random.random() < cfg.anomaly_rate:
            v, c, f = _inject_anomaly(v, c, f)

        yield {"voltage": v, "current": c, "frequency": f}
        t += 1
