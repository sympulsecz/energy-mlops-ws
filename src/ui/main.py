from __future__ import annotations

import time
from collections import deque
import os
import sys
from pathlib import Path
from typing import Tuple
from typing import Dict, List

import requests
import streamlit as st

SIM_SOURCE = "unknown"
SIM_IMPORT_ERROR = ""


def _try_import_sim() -> Tuple[str, object, object, str]:
    """Try several strategies to import a simulator.
    Returns: (source_label, SimConfig, stream_readings, error_text)
    """
    try:
        from src.ui.sim_local import SimConfig as _Cfg, stream_readings as _stream  # type: ignore

        return "ui.sim_local", _Cfg, _stream, ""
    except Exception as e1:  # noqa: F841
        err1 = str(e1)
    try:
        from src.backend.sim import SimConfig as _Cfg, stream_readings as _stream  # type: ignore

        return "backend.sim", _Cfg, _stream, ""
    except Exception as e2:  # noqa: F841
        err2 = str(e2)
    try:
        here = Path(__file__).resolve()
        src_root = str(here.parent.parent)
        if src_root not in sys.path:
            sys.path.append(src_root)
        from ui.sim_local import SimConfig as _Cfg, stream_readings as _stream  # type: ignore

        return "ui.sim_local(sys.path)", _Cfg, _stream, ""
    except Exception as e3:  # noqa: F841
        err3 = str(e3)
    try:
        import math
        import random
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class _Cfg:  # type: ignore
            voltage_base: float = 230.0
            voltage_noise: float = 3.5
            current_base: float = 10.0
            current_noise: float = 2.0
            frequency_base: float = 50.0
            frequency_noise: float = 0.05
            anomaly_rate: float = 0.02

        def _normal(mu: float, sigma: float) -> float:
            return random.gauss(mu, sigma)

        def _inject_anomaly(v: float, c: float, f: float):
            t = random.choice(
                ["v_spike", "v_dip", "c_spike", "c_dip", "f_spike", "f_dip"]
            )
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

        def _stream(cfg: Optional[_Cfg] = None):  # type: ignore
            cfg = cfg or _Cfg()
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

        return (
            "ui.inline",
            _Cfg,
            _stream,
            f"pkg import errors: {err1} | {err2} | {err3}",
        )
    except Exception as e4:
        return "none", None, None, str(e4)


SIM_SOURCE, SimConfig, stream_readings, SIM_IMPORT_ERROR = _try_import_sim()


DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
WINDOW = 200


def init_state() -> None:
    if "buffer" not in st.session_state:
        st.session_state.buffer = deque(maxlen=WINDOW)  # type: Deque[Dict]
    if "running" not in st.session_state:
        st.session_state.running = False
    if "anomaly_rate" not in st.session_state:
        st.session_state.anomaly_rate = 0.02
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = DEFAULT_BACKEND
    if "sim" not in st.session_state:
        if stream_readings is not None:
            st.session_state.sim = stream_readings(
                SimConfig(anomaly_rate=st.session_state.anomaly_rate)
            )
        else:
            st.session_state.sim = None
    st.session_state.sim_source = SIM_SOURCE


def reset_simulator(anomaly_rate: float) -> None:
    if stream_readings is None:
        return
    st.session_state.sim = stream_readings(SimConfig(anomaly_rate=anomaly_rate))


def fetch_health(url: str) -> Dict:
    try:
        r = requests.get(f"{url}/health", timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def predict(url: str, readings: List[Dict]) -> List[Dict]:
    payload = {"readings": readings}
    r = requests.post(f"{url}/predict", json=payload, timeout=5)
    r.raise_for_status()
    data = r.json()
    preds = data.get("predictions", [])
    return preds


def take_batch(batch_size: int) -> List[Dict]:
    sims: List[Dict] = []
    sim = st.session_state.sim
    if sim is None:
        sims = [
            {"voltage": 230.0, "current": 10.0, "frequency": 50.0}
            for _ in range(batch_size)
        ]
        return sims
    for _ in range(batch_size):
        sims.append(next(sim))
    return sims


def update_buffer(readings: List[Dict], preds: List[Dict]) -> None:
    for r, p in zip(readings, preds):
        item = {
            **r,
            "anomaly": bool(p.get("anomaly", False)),
            "score": float(p.get("score", 0.0)),
        }
        st.session_state.buffer.append(item)


def render_chart() -> None:
    if not st.session_state.buffer:
        st.info("No data yet. Generate a batch to begin.")
        return
    vs, cs, fs, anom = [], [], [], []
    for x in st.session_state.buffer:
        vs.append(x["voltage"])
        cs.append(x["current"])
        fs.append(x["frequency"])
        anom.append(x["voltage"] if x["anomaly"] else None)

    st.subheader("Sensor Readings (last {} points)".format(len(vs)))
    st.line_chart({"voltage": vs, "current": cs, "frequency": fs})
    st.caption("Anomalies are highlighted below as markers on voltage.")
    st.line_chart({"anomaly_voltage": anom})


def inject_extreme_reading() -> Tuple[Dict, Dict]:
    reading = {"voltage": 275.0, "current": 28.0, "frequency": 50.7}
    try:
        preds = predict(st.session_state.backend_url, [reading])
        pred = preds[0] if preds else {"anomaly": False, "score": 0.0}
    except Exception:
        pred = {"anomaly": False, "score": 0.0}
    return reading, pred


def main() -> None:
    st.set_page_config(page_title="Grid Anomaly Detection — UI", layout="wide")
    init_state()

    with st.sidebar:
        st.header("Controls")
        backend_url = st.text_input("Backend URL", st.session_state.backend_url)
        if backend_url != st.session_state.backend_url:
            st.session_state.backend_url = backend_url

        batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32)
        interval = st.slider("Interval (s)", 0.1, 5.0, 0.5)
        anomaly_rate = st.slider(
            "Anomaly rate", 0.0, 0.2, float(st.session_state.anomaly_rate), 0.01
        )
        if anomaly_rate != st.session_state.anomaly_rate:
            st.session_state.anomaly_rate = anomaly_rate
            reset_simulator(anomaly_rate)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Generate Batch"):
                readings = take_batch(int(batch_size))
                try:
                    preds = predict(st.session_state.backend_url, readings)
                    update_buffer(readings, preds)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with col_b:
            if st.session_state.running:
                if st.button("Stop Streaming"):
                    st.session_state.running = False
            else:
                if st.button("Start Streaming"):
                    st.session_state.running = True

        if st.button("Inject Extreme Reading"):
            reading, pred = inject_extreme_reading()
            update_buffer([reading], [pred])

    st.title("Real-Time Grid Anomaly Detection — Demo UI")
    health = fetch_health(st.session_state.backend_url)
    cols = st.columns(3)
    with cols[0]:
        if "error" in health:
            st.error("Backend unreachable")
            st.caption(health["error"])
        else:
            st.success("Backend healthy")
            st.caption(
                f"Model: {health.get('model')} — Features: {health.get('features')}"
            )
    with cols[1]:
        total = len(st.session_state.buffer)
        st.metric("Total points", total)
    with cols[2]:
        anomalies = sum(1 for x in st.session_state.buffer if x.get("anomaly"))
        st.metric("Anomalies", anomalies)
    st.caption(
        f"Simulator source: {st.session_state.sim_source} — Anomaly rate: {st.session_state.anomaly_rate}"
    )
    if SIM_IMPORT_ERROR and st.session_state.sim_source.startswith("ui.inline"):
        st.warning(f"Simulator inline fallback in use: {SIM_IMPORT_ERROR}")

    render_chart()

    if st.session_state.running:
        try:
            readings = take_batch(int(batch_size))
            preds = predict(st.session_state.backend_url, readings)
            update_buffer(readings, preds)
        except Exception as e:
            st.warning(f"Streaming error: {e}")
            st.session_state.running = False
        time.sleep(float(interval))
        st.rerun()


if __name__ == "__main__":
    main()
