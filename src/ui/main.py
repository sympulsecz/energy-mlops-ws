from __future__ import annotations

import time
from collections import deque
import os
import sys
from pathlib import Path
from typing import Tuple
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

import requests
import streamlit as st
import pandas as pd
import altair as alt

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
    if "new_conn_each_request" not in st.session_state:
        st.session_state.new_conn_each_request = True
    if "burst_reqs" not in st.session_state:
        st.session_state.burst_reqs = 200
    if "burst_workers" not in st.session_state:
        st.session_state.burst_workers = 32
    if "burst_cycles" not in st.session_state:
        st.session_state.burst_cycles = 5
    if "burst_pause" not in st.session_state:
        st.session_state.burst_pause = 0.2
    if "last_summary" not in st.session_state:
        st.session_state.last_summary = {}


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


def predict(url: str, readings: List[Dict]) -> Tuple[List[Dict], str, float]:
    payload = {"readings": readings}
    headers = {"Connection": "close"} if st.session_state.get("new_conn_each_request") else {}
    t0 = time.perf_counter()
    r = requests.post(f"{url}/predict", json=payload, headers=headers, timeout=5)
    r.raise_for_status()
    data = r.json()
    preds = data.get("predictions", [])
    served_by = data.get("served_by", "unknown")
    dt = time.perf_counter() - t0
    return preds, served_by, dt


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


def update_buffer(readings: List[Dict], preds: List[Dict], served_by: str = "") -> None:
    for r, p in zip(readings, preds):
        item = {
            **r,
            "anomaly": bool(p.get("anomaly", False)),
            "score": float(p.get("score", 0.0)),
            "served_by": served_by,
        }
        st.session_state.buffer.append(item)


def _latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg_s": 0.0, "p95_s": 0.0, "p99_s": 0.0}
    lats = sorted(latencies)
    n = len(lats)
    p95 = lats[min(max(int(0.95 * n) - 1, 0), n - 1)]
    p99 = lats[min(max(int(0.99 * n) - 1, 0), n - 1)]
    return {"avg_s": round(mean(lats), 4), "p95_s": round(p95, 4), "p99_s": round(p99, 4)}


def set_summary(action: str, ok: int, err: int, latencies: List[float], anomalies: int, served_counts: Dict[str, int]) -> None:
    stats = _latency_stats(latencies)
    distinct = len(served_counts)
    top_inst = max(served_counts.items(), key=lambda x: x[1])[0] if served_counts else ""
    st.session_state.last_summary = {
        "action": action,
        "requests_ok": ok,
        "requests_err": err,
        "anomalies": anomalies,
        "latency_avg_s": stats["avg_s"],
        "latency_p95_s": stats["p95_s"],
        "latency_p99_s": stats["p99_s"],
        "instances": distinct,
        "top_instance": top_inst,
    }


def render_chart() -> None:
    if not st.session_state.buffer:
        st.info("No data yet. Generate a batch to begin.")
        return
    rows = []
    for i, x in enumerate(st.session_state.buffer):
        rows.append(
            {
                "idx": i,
                "voltage": x["voltage"],
                "current": x["current"],
                "frequency": x["frequency"],
                "anomaly": bool(x.get("anomaly", False)),
                "score": float(x.get("score", 0.0)),
                "served_by": str(x.get("served_by", "")),
            }
        )
    df = pd.DataFrame(rows)

    st.subheader(f"Sensor Readings (last {len(df)} points)")
    df_long = df.melt(id_vars=["idx", "anomaly", "score", "served_by"], value_vars=["voltage", "current", "frequency"], var_name="series", value_name="value")

    base = alt.Chart(df_long).mark_line().encode(
        x=alt.X("idx:Q", title="time (index)"),
        y=alt.Y("value:Q", title="value"),
        color=alt.Color("series:N", legend=alt.Legend(title="signal")),
        tooltip=["series", alt.Tooltip("value:Q", format=".2f"), "idx"],
    )

    anom_points = (
        alt.Chart(df[df["anomaly"]])
        .mark_point(color="#ff4b4b", size=80, shape="triangle-up")
        .encode(
            x="idx:Q",
            y=alt.Y("voltage:Q", title="value"),
            tooltip=[
                alt.Tooltip("voltage:Q", title="voltage", format=".2f"),
                alt.Tooltip("current:Q", title="current", format=".2f"),
                alt.Tooltip("frequency:Q", title="frequency", format=".3f"),
                alt.Tooltip("score:Q", title="anomaly score", format=".3f"),
                alt.Tooltip("served_by:N", title="served by"),
                alt.Tooltip("idx:Q", title="index"),
            ],
        )
    )

    chart = (base + anom_points).properties(height=280).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption("Red triangles mark anomaly points on the voltage series.")

    served_counts: Dict[str, int] = {}
    for x in st.session_state.buffer:
        key = str(x.get("served_by", ""))
        if not key:
            continue
        served_counts[key] = served_counts.get(key, 0) + 1
    if served_counts:
        st.subheader("Requests Served by Instance (last window)")
        st.bar_chart(served_counts)


def inject_extreme_reading() -> Tuple[Dict, Dict, str]:
    reading = {"voltage": 275.0, "current": 28.0, "frequency": 50.7}
    try:
        preds, served_by, _ = predict(st.session_state.backend_url, [reading])
        pred = preds[0] if preds else {"anomaly": False, "score": 0.0}
    except Exception:
        pred = {"anomaly": False, "score": 0.0}
        served_by = ""
    return reading, pred, served_by


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

        st.checkbox(
            "New connection each request (improves load spread)",
            key="new_conn_each_request",
            value=st.session_state.new_conn_each_request,
        )
        st.number_input(
            "Burst requests (N)",
            min_value=1,
            max_value=5000,
            value=int(st.session_state.burst_reqs),
            key="burst_reqs",
        )
        st.number_input(
            "Concurrent workers",
            min_value=1,
            max_value=256,
            value=int(st.session_state.burst_workers),
            key="burst_workers",
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.number_input(
                "Burst cycles",
                min_value=1,
                max_value=100,
                value=int(st.session_state.burst_cycles),
                key="burst_cycles",
            )
        with col_c2:
            st.number_input(
                "Pause between cycles (s)",
                min_value=0.0,
                max_value=5.0,
                value=float(st.session_state.burst_pause),
                step=0.1,
                key="burst_pause",
            )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Generate Batch"):
                readings = take_batch(int(batch_size))
                try:
                    preds, served_by, dt = predict(st.session_state.backend_url, readings)
                    update_buffer(readings, preds, served_by)
                    anomalies_now = sum(1 for p in preds if bool(p.get("anomaly", False)))
                    set_summary(
                        "generate_batch",
                        ok=1,
                        err=0,
                        latencies=[dt],
                        anomalies=anomalies_now,
                        served_counts={served_by: 1} if served_by else {},
                    )
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
            reading, pred, served_by = inject_extreme_reading()
            update_buffer([reading], [pred], served_by)
            set_summary(
                "inject_extreme",
                ok=1,
                err=0,
                latencies=[],
                anomalies=int(bool(pred.get("anomaly", False))),
                served_counts={served_by: 1} if served_by else {},
            )

        if st.button("Burst Load"):
            try:
                latencies: List[float] = []
                ok = 0
                err = 0
                anomalies_total = 0
                served_counts: Dict[str, int] = {}
                for _ in range(int(st.session_state.burst_reqs)):
                    readings = take_batch(int(batch_size))
                    preds, served_by, dt = predict(
                        st.session_state.backend_url, readings
                    )
                    update_buffer(readings, preds, served_by)
                    latencies.append(dt)
                    ok += 1
                    anomalies_total += sum(1 for p in preds if bool(p.get("anomaly", False)))
                    served_counts[served_by] = served_counts.get(served_by, 0) + 1
                st.success("Burst completed")
                set_summary(
                    "burst_seq",
                    ok=ok,
                    err=err,
                    latencies=latencies,
                    anomalies=anomalies_total,
                    served_counts=served_counts,
                )
            except Exception as e:
                st.error(f"Burst failed: {e}")

        if st.button("Concurrent Burst"):
            try:
                total = int(st.session_state.burst_reqs)
                workers = int(st.session_state.burst_workers)
                progress = st.progress(0.0, text="Dispatching concurrent requests…")

                batches: List[List[Dict]] = [take_batch(int(batch_size)) for _ in range(total)]
                headers = {"Connection": "close"} if st.session_state.get("new_conn_each_request") else {}
                backend_url = st.session_state.backend_url

                def submit_one(rs: List[Dict]):
                    t0 = time.perf_counter()
                    r = requests.post(
                        f"{backend_url}/predict",
                        json={"readings": rs},
                        headers=headers,
                        timeout=5,
                    )
                    r.raise_for_status()
                    dt = time.perf_counter() - t0
                    data = r.json()
                    return data.get("predictions", []), data.get("served_by", "unknown"), dt, rs

                done = 0
                latencies: List[float] = []
                ok = 0
                err = 0
                anomalies_total = 0
                served_counts: Dict[str, int] = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(submit_one, rs) for rs in batches]
                    for fut in as_completed(futures):
                        preds, served_by, dt, rs = fut.result()
                        update_buffer(rs, preds, served_by)
                        latencies.append(dt)
                        ok += 1
                        anomalies_total += sum(1 for p in preds if bool(p.get("anomaly", False)))
                        served_counts[served_by] = served_counts.get(served_by, 0) + 1
                        done += 1
                        progress.progress(min(done / total, 1.0))
                progress.empty()
                st.success(f"Concurrent burst completed: {done} requests")
                set_summary(
                    "burst_concurrent",
                    ok=ok,
                    err=err,
                    latencies=latencies,
                    anomalies=anomalies_total,
                    served_counts=served_counts,
                )
            except Exception as e:
                st.error(f"Concurrent burst failed: {e}")

        if st.button("Sustained Concurrent Bursts"):
            try:
                cycles = int(st.session_state.burst_cycles)
                pause = float(st.session_state.burst_pause)
                total = int(st.session_state.burst_reqs)
                workers = int(st.session_state.burst_workers)
                overall = st.progress(0.0, text="Running sustained bursts…")
                for c in range(cycles):
                    batches: List[List[Dict]] = [
                        take_batch(int(batch_size)) for _ in range(total)
                    ]
                    headers = (
                        {"Connection": "close"}
                        if st.session_state.get("new_conn_each_request")
                        else {}
                    )
                    backend_url = st.session_state.backend_url

                    def submit_one(rs: List[Dict]):
                        t0 = time.perf_counter()
                        r = requests.post(
                            f"{backend_url}/predict",
                            json={"readings": rs},
                            headers=headers,
                            timeout=5,
                        )
                        r.raise_for_status()
                        dt = time.perf_counter() - t0
                        data = r.json()
                        return (
                            data.get("predictions", []),
                            data.get("served_by", "unknown"),
                            dt,
                            rs,
                        )

                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(submit_one, rs) for rs in batches]
                        for fut in as_completed(futures):
                            preds, served_by, dt, rs = fut.result()
                            update_buffer(rs, preds, served_by)
                            latencies.append(dt)
                            ok += 1
                            anomalies_total += sum(1 for p in preds if bool(p.get("anomaly", False)))
                            served_counts[served_by] = served_counts.get(served_by, 0) + 1
                    overall.progress(min((c + 1) / cycles, 1.0))
                    if c + 1 < cycles and pause > 0:
                        time.sleep(pause)
                overall.empty()
                st.success(
                    f"Sustained bursts completed: {cycles} cycles x {total} requests"
                )
                set_summary(
                    "burst_sustained",
                    ok=ok,
                    err=err,
                    latencies=latencies,
                    anomalies=anomalies_total,
                    served_counts=served_counts,
                )
            except Exception as e:
                st.error(f"Sustained bursts failed: {e}")

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

    if st.session_state.last_summary:
        st.subheader("Request Summary (last action)")
        summary_df = pd.DataFrame(
            list(st.session_state.last_summary.items()), columns=["metric", "value"]
        )
        st.table(summary_df)

    if st.session_state.running:
        try:
            readings = take_batch(int(batch_size))
            preds, served_by, dt = predict(st.session_state.backend_url, readings)
            update_buffer(readings, preds, served_by)
            anomalies_now = sum(1 for p in preds if bool(p.get("anomaly", False)))
            set_summary(
                "stream_tick",
                ok=1,
                err=0,
                latencies=[dt],
                anomalies=anomalies_now,
                served_counts={served_by: 1} if served_by else {},
            )
        except Exception as e:
            st.warning(f"Streaming error: {e}")
            st.session_state.running = False
        time.sleep(float(interval))
        st.rerun()


if __name__ == "__main__":
    main()
