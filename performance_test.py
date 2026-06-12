"""
Comprehensive performance test for Academic Shield (deployed or local).

Tests:
  1. Connectivity & DNS
  2. HTTP response metrics (TTFB, total, content size)
  3. Latency distribution (min/mean/median/p95/p99/max)
  4. Concurrent user simulation (load test)
  5. Sustained load (throughput & error rate)
  6. Local model inference latency
  7. Full report with pass/fail verdict

Usage:
    # Test deployed app
    python performance_test.py --url https://yourapp.vercel.app

    # Test local app
    python performance_test.py --url http://127.0.0.1:8123

    # Heavy load test
    python performance_test.py --url https://yourapp.vercel.app --n 200 --users 20

Requirements:
    pip install requests
"""

import argparse
import os
import socket
import statistics
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Missing dependency. Run: pip install requests")
    sys.exit(1)

# ── thresholds (ms) ───────────────────────────────────────────────────────────
TARGETS = {
    "ttfb_mean": 500,  # Time To First Byte mean
    "total_mean": 2000,  # Total response time mean
    "total_p95": 3000,  # 95th percentile response
    "error_rate_pct": 5,  # Max acceptable error %
    "inference_mean": 100,  # Local model inference (A+B combined)
    "concurrent_mean": 3000,  # Mean under concurrent load
}

# ── color output ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):
    return f"{GREEN}✅ PASS{RESET}  {msg}"


def warn(msg):
    return f"{YELLOW}⚠️  WARN{RESET}  {msg}"


def fail(msg):
    return f"{RED}❌ FAIL{RESET}  {msg}"


def info(msg):
    return f"{CYAN}ℹ{RESET}  {msg}"


def verdict(value, target, label, unit="ms", lower_is_better=True):
    ok_thresh = target
    warn_thresh = target * 2
    if lower_is_better:
        if value <= ok_thresh:
            return ok(f"{label}: {value:.1f}{unit}  (target ≤ {target}{unit})")
        elif value <= warn_thresh:
            return warn(f"{label}: {value:.1f}{unit}  (target ≤ {target}{unit})")
        else:
            return fail(f"{label}: {value:.1f}{unit}  (target ≤ {target}{unit})")
    else:
        if value >= ok_thresh:
            return ok(f"{label}: {value:.1f}{unit}  (target ≥ {target}{unit})")
        else:
            return fail(f"{label}: {value:.1f}{unit}  (target ≥ {target}{unit})")


# ── data container ────────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    success: bool
    ttfb_ms: float = 0.0
    total_ms: float = 0.0
    status_code: int = 0
    content_bytes: int = 0
    error: str = ""


# ── single request ────────────────────────────────────────────────────────────
def make_request(
    url: str, session: requests.Session, timeout: int = 15
) -> RequestResult:
    try:
        t_start = time.perf_counter()
        with session.get(url, timeout=timeout, stream=True) as resp:
            ttfb = (time.perf_counter() - t_start) * 1000
            content = resp.content
            total = (time.perf_counter() - t_start) * 1000
        return RequestResult(
            success=resp.status_code < 400,
            ttfb_ms=ttfb,
            total_ms=total,
            status_code=resp.status_code,
            content_bytes=len(content),
        )
    except Exception as e:
        total = (time.perf_counter() - t_start) * 1000
        return RequestResult(success=False, total_ms=total, error=str(e)[:80])


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=1, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "AcademicShield-PerfTest/1.0"})
    return s


# ── stats helper ──────────────────────────────────────────────────────────────
def compute_stats(values: list[float]) -> dict:
    if not values:
        return {}
    sv = sorted(values)
    n = len(sv)
    return {
        "n": n,
        "mean": statistics.mean(sv),
        "median": statistics.median(sv),
        "std": statistics.stdev(sv) if n > 1 else 0.0,
        "min": sv[0],
        "max": sv[-1],
        "p75": sv[int(0.75 * n)],
        "p90": sv[int(0.90 * n)],
        "p95": sv[min(int(0.95 * n), n - 1)],
        "p99": sv[min(int(0.99 * n), n - 1)],
    }


def print_stats(label: str, s: dict, unit: str = "ms") -> None:
    if not s:
        print(f"  {label}: no data")
        return
    print(f"\n  {BOLD}{label}{RESET}")
    print(f"  {'Runs':<12}: {s['n']}")
    print(f"  {'Mean':<12}: {s['mean']:>8.1f} {unit}  ← primary metric")
    print(f"  {'Median':<12}: {s['median']:>8.1f} {unit}")
    print(f"  {'Std dev':<12}: {s['std']:>8.1f} {unit}")
    print(f"  {'Min':<12}: {s['min']:>8.1f} {unit}")
    print(f"  {'Max':<12}: {s['max']:>8.1f} {unit}")
    print(f"  {'p75':<12}: {s['p75']:>8.1f} {unit}")
    print(f"  {'p90':<12}: {s['p90']:>8.1f} {unit}")
    print(f"  {'p95':<12}: {s['p95']:>8.1f} {unit}  ← 95% of users ≤ this")
    print(f"  {'p99':<12}: {s['p99']:>8.1f} {unit}")


# ── section 1: DNS + connectivity ────────────────────────────────────────────
def test_connectivity(url: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    results = {}

    # DNS
    t0 = time.perf_counter()
    try:
        ip = socket.gethostbyname(host)
        results["dns_ms"] = (time.perf_counter() - t0) * 1000
        results["dns_ip"] = ip
        results["dns_ok"] = True
    except Exception as e:
        results["dns_ms"] = (time.perf_counter() - t0) * 1000
        results["dns_ok"] = False
        results["dns_err"] = str(e)
        return results

    # TCP connect
    t0 = time.perf_counter()
    try:
        sock = socket.create_connection((host, port), timeout=5)
        results["tcp_ms"] = (time.perf_counter() - t0) * 1000
        results["tcp_ok"] = True
        sock.close()
    except Exception as e:
        results["tcp_ms"] = (time.perf_counter() - t0) * 1000
        results["tcp_ok"] = False
        results["tcp_err"] = str(e)

    return results


# ── section 2: sequential latency ────────────────────────────────────────────
def test_sequential(url: str, n: int, warmup: int) -> tuple[list, list, list]:
    session = make_session()
    print(f"  Warming up ({warmup} requests)...", end="", flush=True)
    for _ in range(warmup):
        make_request(url, session)
    print(" done")

    results, ttfbs, totals = [], [], []
    print(f"  Running {n} sequential requests...", end="", flush=True)
    for i in range(n):
        r = make_request(url, session)
        results.append(r)
        if r.success:
            ttfbs.append(r.ttfb_ms)
            totals.append(r.total_ms)
        if (i + 1) % 20 == 0:
            print(f" {i+1}", end="", flush=True)
    print(" done")
    return results, ttfbs, totals


# ── section 3: concurrent load test ──────────────────────────────────────────
def test_concurrent(
    url: str, n_users: int, n_requests: int
) -> tuple[list[RequestResult], float]:
    results = []
    lock = threading.Lock()
    sessions = [make_session() for _ in range(n_users)]

    def worker(session_idx: int, req_count: int):
        local = []
        for _ in range(req_count):
            r = make_request(url, sessions[session_idx])
            local.append(r)
        with lock:
            results.extend(local)

    requests_per_user = max(1, n_requests // n_users)
    print(
        f"  Simulating {n_users} concurrent users × {requests_per_user} req each...",
        end="",
        flush=True,
    )

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_users) as ex:
        futs = [ex.submit(worker, i, requests_per_user) for i in range(n_users)]
        for f in as_completed(futs):
            f.result()
    elapsed = time.perf_counter() - t_start

    print(f" done ({elapsed:.1f}s)")
    return results, elapsed


# ── section 4: sustained throughput ──────────────────────────────────────────
def test_throughput(url: str, duration_s: int, n_users: int) -> dict:
    results = []
    lock = threading.Lock()
    stop_evt = threading.Event()

    def worker():
        s = make_session()
        while not stop_evt.is_set():
            r = make_request(url, s)
            with lock:
                results.append(r)

    print(f"  Sustained {n_users}-user load for {duration_s}s...", end="", flush=True)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_users)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop_evt.set()
    for t in threads:
        t.join(timeout=5)
    elapsed = time.perf_counter() - t_start
    print(f" done ({len(results)} requests in {elapsed:.1f}s)")

    successes = [r for r in results if r.success]
    return {
        "total_requests": len(results),
        "success": len(successes),
        "errors": len(results) - len(successes),
        "error_rate_pct": (len(results) - len(successes)) / max(1, len(results)) * 100,
        "rps": len(results) / elapsed,
        "totals": [r.total_ms for r in successes],
        "elapsed_s": elapsed,
    }


# ── section 5: local inference ───────────────────────────────────────────────
def test_local_inference(n: int, warmup: int) -> dict | None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import joblib
        import numpy as np
        import pandas as pd

        try:
            from src.features import patch_main, engineer_features_a
            from src.config import (
                STRESS_MAPPING_A,
                STRESS_MAPPING_B,
                MODEL_A_BASE_FEATURES,
                MODEL_B_BASE_FEATURES,
                BURNOUT_SCORE_WEIGHTS,
            )

            patch_main()
        except Exception:
            STRESS_MAPPING_A = {"Low": 2.29, "Moderate": 4.80, "High": 7.42}
            STRESS_MAPPING_B = {"Low": 0, "Moderate": 1, "High": 2}
            MODEL_A_BASE_FEATURES = [
                "study_hours_per_day",
                "sleep_hours",
                "exam_pressure",
                "stress_level",
                "financial_stress",
                "social_support",
                "anxiety_score",
                "depression_score",
                "family_expectation",
                "physical_activity",
            ]
            MODEL_B_BASE_FEATURES = [
                "study_hours",
                "eca_hours",
                "sleep_hours",
                "social_hours",
                "physical_hours",
                "stress_level",
            ]
            BURNOUT_SCORE_WEIGHTS = [20.0, 55.0, 85.0]

            def engineer_features_a(df):
                df = df.copy()
                df["stress_x_anxiety"] = df["stress_level"] * df["anxiety_score"]
                df["stress_x_depression"] = df["stress_level"] * df["depression_score"]
                df["stress_x_exam"] = df["stress_level"] * df["exam_pressure"]
                df["anxiety_x_depression"] = (
                    df["anxiety_score"] * df["depression_score"]
                )
                df["study_sleep_ratio"] = df["study_hours_per_day"] / (
                    df["sleep_hours"] + 1
                )
                df["pressure_support_gap"] = (
                    (
                        df["exam_pressure"]
                        + df["financial_stress"]
                        + df["family_expectation"]
                    )
                    / 3
                ) - df["social_support"]
                df["stress_level_sq"] = df["stress_level"] ** 2
                df["anxiety_score_sq"] = df["anxiety_score"] ** 2
                df["sleep_deprived"] = (df["sleep_hours"] < 5).astype(int)
                df["high_stress"] = (df["stress_level"] > 7).astype(int)
                return df

        print("  Loading models...", end="", flush=True)
        t0 = time.perf_counter()
        mA = joblib.load("models/modelA.pkl")
        mB = joblib.load("models/modelB.pkl")
        load_ms = (time.perf_counter() - t0) * 1000
        print(f" done ({load_ms:.0f} ms)")

        SAMPLE = dict(
            zip(
                MODEL_A_BASE_FEATURES,
                [
                    6.0,
                    7.0,
                    6.0,
                    STRESS_MAPPING_A["Moderate"],
                    5.0,
                    5.0,
                    4.0,
                    3.0,
                    6.0,
                    1.5,
                ],
            )
        )

        def infer_a():
            df = pd.DataFrame([SAMPLE], columns=MODEL_A_BASE_FEATURES)
            t0 = time.perf_counter()
            eng = engineer_features_a(df)
            _ = mA.predict(eng)
            prb = mA.predict_proba(eng)
            _ = float(np.dot(prb[0], BURNOUT_SCORE_WEIGHTS))
            return (time.perf_counter() - t0) * 1000

        def infer_b():
            stress = STRESS_MAPPING_B["Moderate"]
            row = dict(zip(MODEL_B_BASE_FEATURES, [6.0, 2.0, 7.0, 3.0, 1.5, stress]))
            df = pd.DataFrame([row], columns=MODEL_B_BASE_FEATURES)
            t0 = time.perf_counter()
            _ = mB.predict(df)
            return (time.perf_counter() - t0) * 1000

        for _ in range(warmup):
            infer_a()
            infer_b()

        a_times = [infer_a() for _ in range(n)]
        b_times = [infer_b() for _ in range(n)]
        combined = [a + b for a, b in zip(a_times, b_times)]

        return {
            "load_ms": load_ms,
            "a_times": a_times,
            "b_times": b_times,
            "combined": combined,
        }
    except Exception as e:
        print(f"\n  [skip] Could not run local inference: {e}")
        return None


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Academic Shield — Performance Test")
    parser.add_argument("--url", default="http://localhost:8123", help="App URL")
    parser.add_argument(
        "--n", type=int, default=50, help="Sequential requests (default: 50)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup requests (default: 3)"
    )
    parser.add_argument(
        "--users", type=int, default=10, help="Concurrent users (default: 10)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Sustained load duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--infer-n", type=int, default=100, help="Local inference runs (default: 100)"
    )
    parser.add_argument(
        "--skip-http", action="store_true", help="Skip HTTP tests (inference only)"
    )
    parser.add_argument(
        "--skip-infer", action="store_true", help="Skip local inference test"
    )
    args = parser.parse_args()

    W = 60
    print(f"\n{'═'*W}")
    print(f"  {BOLD}Academic Shield — Performance Test{RESET}")
    print(f"  Target URL : {args.url}")
    print(f"  Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*W}")

    verdicts = []

    # ── 1. Connectivity ───────────────────────────────────────────────────────
    if not args.skip_http:
        print(f"\n{BOLD}[1] CONNECTIVITY{RESET}")
        conn = test_connectivity(args.url)
        if conn.get("dns_ok"):
            dns_v = verdict(conn["dns_ms"], 200, "DNS resolution", lower_is_better=True)
            print(f"  DNS resolved: {conn['dns_ip']}")
            print(f"  {dns_v}")
            verdicts.append(("DNS", conn["dns_ms"] <= 200))
        else:
            print(fail(f"DNS failed: {conn.get('dns_err','')}"))
            verdicts.append(("DNS", False))

        if conn.get("tcp_ok"):
            tcp_v = verdict(conn["tcp_ms"], 300, "TCP connect", lower_is_better=True)
            print(f"  {tcp_v}")
            verdicts.append(("TCP", conn["tcp_ms"] <= 300))
        elif "tcp_err" in conn:
            print(fail(f"TCP failed: {conn['tcp_err']}"))
            verdicts.append(("TCP", False))

    # ── 2. Sequential latency ─────────────────────────────────────────────────
    if not args.skip_http:
        print(f"\n{BOLD}[2] SEQUENTIAL LATENCY ({args.n} requests){RESET}")
        seq_results, ttfbs, totals = test_sequential(args.url, args.n, args.warmup)

        errors = [r for r in seq_results if not r.success]
        error_rate = len(errors) / len(seq_results) * 100
        avg_bytes = (
            statistics.mean([r.content_bytes for r in seq_results if r.success])
            if totals
            else 0
        )

        print(
            f"\n  Requests   : {len(seq_results)}  success={len(seq_results)-len(errors)}  errors={len(errors)}"
        )
        print(f"  Error rate : {error_rate:.1f}%")
        print(f"  Avg size   : {avg_bytes/1024:.1f} KB")
        if errors:
            codes = set(r.status_code for r in errors)
            msgs = set(r.error for r in errors if r.error)
            print(f"  Error codes: {codes}")
            if msgs:
                print(f"  Errors     : {list(msgs)[:3]}")

        if ttfbs:
            ts = compute_stats(ttfbs)
            print_stats("Time To First Byte (TTFB)", ts)
            verdicts.append(("TTFB mean", ts["mean"] <= TARGETS["ttfb_mean"]))

        if totals:
            ts2 = compute_stats(totals)
            print_stats("Total Response Time", ts2)
            verdicts.append(("Response mean", ts2["mean"] <= TARGETS["total_mean"]))
            verdicts.append(("Response p95", ts2["p95"] <= TARGETS["total_p95"]))

        verdicts.append(("Error rate", error_rate <= TARGETS["error_rate_pct"]))

    # ── 3. Concurrent load ────────────────────────────────────────────────────
    if not args.skip_http:
        print(f"\n{BOLD}[3] CONCURRENT LOAD ({args.users} users){RESET}")
        conc_results, elapsed = test_concurrent(args.url, args.users, args.users * 5)

        c_success = [r for r in conc_results if r.success]
        c_errors = len(conc_results) - len(c_success)
        c_err_pct = c_errors / max(1, len(conc_results)) * 100
        c_totals = [r.total_ms for r in c_success]
        c_rps = len(conc_results) / elapsed

        if c_totals:
            cs = compute_stats(c_totals)
            print_stats(f"Concurrent Response Time ({args.users} users)", cs)
            print(f"\n  Throughput : {c_rps:.1f} req/s")
            print(f"  Errors     : {c_errors}/{len(conc_results)}  ({c_err_pct:.1f}%)")
            verdicts.append(
                ("Concurrent mean", cs["mean"] <= TARGETS["concurrent_mean"])
            )
            verdicts.append(
                ("Concurrent errors", c_err_pct <= TARGETS["error_rate_pct"])
            )

    # ── 4. Sustained throughput ───────────────────────────────────────────────
    if not args.skip_http:
        print(
            f"\n{BOLD}[4] SUSTAINED LOAD ({args.users} users × {args.duration}s){RESET}"
        )
        tput = test_throughput(args.url, args.duration, min(args.users, 5))

        print(f"\n  Total requests : {tput['total_requests']}")
        print(f"  Successes      : {tput['success']}")
        print(f"  Errors         : {tput['errors']}  ({tput['error_rate_pct']:.1f}%)")
        print(f"  Throughput     : {tput['rps']:.2f} req/s")

        if tput["totals"]:
            ts3 = compute_stats(tput["totals"])
            print_stats("Sustained Response Time", ts3)

        verdicts.append(
            ("Sustained errors", tput["error_rate_pct"] <= TARGETS["error_rate_pct"])
        )
        verdicts.append(("Throughput", tput["rps"] >= 0.5))  # at least 0.5 req/s

    # ── 5. Local inference ────────────────────────────────────────────────────
    if not args.skip_infer:
        print(f"\n{BOLD}[5] LOCAL MODEL INFERENCE LATENCY ({args.infer_n} runs){RESET}")
        inf = test_local_inference(args.infer_n, warmup=5)
        if inf:
            print(f"\n  Model load (cold)   : {inf['load_ms']:.0f} ms")
            print(
                f"  Note: cold load happens once per process; subsequent calls are warm"
            )

            print_stats(
                "Model A  (engineer + predict + predict_proba)",
                compute_stats(inf["a_times"]),
            )
            print_stats("Model B  (predict)", compute_stats(inf["b_times"]))
            print_stats(
                "Combined A+B  ← actual per-request inference cost",
                compute_stats(inf["combined"]),
            )

            inf_mean = statistics.mean(inf["combined"])
            verdicts.append(
                ("Inference < 100ms", inf_mean <= TARGETS["inference_mean"])
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  {BOLD}SUMMARY — PASS/FAIL{RESET}")
    print(f"{'═'*W}")

    passed = sum(1 for _, v in verdicts if v)
    total = len(verdicts)

    for label, result in verdicts:
        if result:
            print(f"  {ok(label)}")
        else:
            print(f"  {fail(label)}")

    print(f"\n  Score: {passed}/{total} checks passed")

    if passed == total:
        print(f"\n  {GREEN}{BOLD}🎉 ALL CHECKS PASSED — App is performant!{RESET}")
    elif passed >= total * 0.8:
        print(f"\n  {YELLOW}{BOLD}⚠️  MOSTLY PASSING — Minor issues to address.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}❌ PERFORMANCE ISSUES DETECTED{RESET}")

    print(f"\n  Reference targets:")
    for k, v in TARGETS.items():
        print(f"    {k:<22}: {v} {'ms' if 'ms' not in k else ''}")
    print()


if __name__ == "__main__":
    main()