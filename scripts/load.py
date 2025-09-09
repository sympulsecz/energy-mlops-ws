import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Dict, List, Tuple

import requests


def gen_reading(anomaly_rate: float = 0.02) -> Dict[str, float]:
    v = random.gauss(230.0, 3.5)
    c = random.gauss(10.0, 2.0)
    f = random.gauss(50.0, 0.05)
    if random.random() < anomaly_rate:
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
    return {"voltage": v, "current": c, "frequency": f}


def make_payload(
    batch_size: int, anomaly_rate: float
) -> Dict[str, List[Dict[str, float]]]:
    return {"readings": [gen_reading(anomaly_rate) for _ in range(batch_size)]}


def do_request(
    url: str, payload: Dict, no_keepalive: bool = False
) -> Tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        headers = {"Connection": "close"} if no_keepalive else {}
        r = requests.post(f"{url}/predict", json=payload, headers=headers, timeout=5)
        r.raise_for_status()
        _ = r.json()
        dt = time.perf_counter() - t0
        return True, dt
    except Exception:
        dt = time.perf_counter() - t0
        return False, dt


def run_load(
    url: str,
    rps: float,
    duration: float,
    batch_size: int,
    workers: int,
    anomaly_rate: float,
    no_keepalive: bool,
) -> None:
    interval = 1.0 / rps if rps > 0 else 0
    deadline = time.time() + duration
    latencies: List[float] = []
    ok = 0
    err = 0
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        next_tick = time.time()
        while time.time() < deadline:
            payload = make_payload(batch_size, anomaly_rate)
            futures.append(pool.submit(do_request, url, payload, no_keepalive))
            next_tick += interval
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

        for fut in as_completed(futures):
            success, dt = fut.result()
            latencies.append(dt)
            if success:
                ok += 1
            else:
                err += 1

    if latencies:
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1]
        p99 = latencies_sorted[int(0.99 * len(latencies_sorted)) - 1]
        print(
            json.dumps(
                {
                    "requests_ok": ok,
                    "requests_err": err,
                    "duration_s": duration,
                    "rps_target": rps,
                    "batch_size": batch_size,
                    "latency_avg_s": round(mean(latencies), 4),
                    "latency_p95_s": round(p95, 4),
                    "latency_p99_s": round(p99, 4),
                },
                indent=2,
            )
        )
    else:
        print("No results recorded.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Load generator for anomaly backend")
    ap.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Backend base URL (no trailing slash)",
    )
    ap.add_argument(
        "--rps", type=float, default=50.0, help="Target requests per second"
    )
    ap.add_argument(
        "--duration", type=float, default=60.0, help="Test duration in seconds"
    )
    ap.add_argument("--batch-size", type=int, default=32, help="Readings per request")
    ap.add_argument(
        "--workers", type=int, default=8, help="Max concurrent worker threads"
    )
    ap.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.02,
        help="Probability of anomaly per reading",
    )
    ap.add_argument(
        "--no-keepalive",
        action="store_true",
        help="Close HTTP connection each request to improve load distribution across pods",
    )
    args = ap.parse_args()

    print(
        f"Starting load: url={args.url}, rps={args.rps}, duration={args.duration}s, "
        f"batch={args.batch_size}, workers={args.workers}"
    )
    run_load(
        url=args.url,
        rps=args.rps,
        duration=args.duration,
        batch_size=args.batch_size,
        workers=args.workers,
        anomaly_rate=args.anomaly_rate,
        no_keepalive=args.no_keepalive,
    )


if __name__ == "__main__":
    main()
