"""FASE 23: test de concurrencia controlada contra el backend.

Uso: python scripts/concurrency_test.py [n_requests] [concurrency]
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:5122/api/v1/foodai/analyze"

IMAGES = [
    "datasets/food-us-v0.1/images/train/pizza_001.jpg",
    "datasets/food-us-v0.1/images/train/hot_dog_001.jpg",
    "datasets/food-bench-v1/salmon/img_0001.jpg",
    "datasets/food-bench-v1/nachos/img_0001.jpg",
    "datasets/food-us-v0.1/images/train/french_fries_000.jpg",
    "tests/assets/banana.jpg",
]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TIMEOUT = 120.0

results: list[dict] = []
lock = threading.Lock()


def run_one(i: int) -> None:
    rel = IMAGES[i % len(IMAGES)]
    path = BASE / rel
    t0 = time.perf_counter()
    try:
        with open(path, "rb") as fh:
            resp = requests.post(API, files={"image": (path.name, fh, "image/jpeg")}, timeout=TIMEOUT)
        total = time.perf_counter() - t0
        with lock:
            results.append({"i": i, "status": resp.status_code, "ms": total * 1000,
                            "foods": len(resp.json().get("foods", [])) if resp.status_code == 200 else 0})
    except requests.exceptions.Timeout:
        with lock:
            results.append({"i": i, "status": -1, "ms": TIMEOUT * 1000, "foods": 0, "hang": True})
    except Exception as exc:  # noqa: BLE001
        with lock:
            results.append({"i": i, "status": -2, "ms": 0, "foods": 0, "error": type(exc).__name__})


def main() -> None:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONC) as pool:
        list(pool.map(run_one, range(N)))
    wall = time.perf_counter() - t0

    ok = sum(1 for r in results if r["status"] == 200)
    fail = sum(1 for r in results if r["status"] not in (200,))
    hangs = sum(1 for r in results if r.get("hang"))
    lats = sorted(r["ms"] for r in results)
    foods = sum(r["foods"] for r in results)

    print(f"=== CONCURRENCIA {CONC} — {N} requests ===")
    print(f"ok={ok} fail={fail} hangs={hangs} wall={wall:.1f}s foods_total={foods}")
    if lats:
        n = len(lats)
        print(f"p50={lats[n//2]:.0f}ms p95={lats[int(n*0.95)]:.0f}ms p99={lats[int(n*0.99)]:.0f}ms max={lats[-1]:.0f}ms")
        fast = [r["ms"] for r in results if r["ms"] < 3000]
        slow = [r["ms"] for r in results if r["ms"] >= 3000]
        if fast:
            print(f"fast path: n={len(fast)} p50={sorted(fast)[len(fast)//2]:.0f}ms")
        if slow:
            print(f"dino path: n={len(slow)} p50={sorted(slow)[len(slow)//2]:.0f}ms")


if __name__ == "__main__":
    main()