"""FASE 23: test de estabilidad secuencial contra el backend.

Identifica el request N exacto del primer fallo/hang. Uso:
python scripts/stability_test.py [n_requests] [timeout_s]
"""
import sys
import time
from pathlib import Path

import psutil
import requests

BASE = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:5122/api/v1/foodai/analyze"


def memory_report(tag: str) -> None:
    try:
        food_ai = backend = None
        for conn in psutil.net_connections("tcp"):
            if conn.status == "LISTEN" and conn.laddr:
                if conn.laddr.port == 8010:
                    food_ai = conn.pid
                elif conn.laddr.port == 5122:
                    backend = conn.pid
        parts = [f"{tag}:"]
        for pid, name in ((food_ai, "food-ai"), (backend, "backend")):
            if pid:
                proc = psutil.Process(pid)
                parts.append(f"{name} WS={proc.memory_info().rss // (1024 * 1024)}MB")
        print(" ".join(parts), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{tag}: memoria no medible ({exc})", flush=True)

IMAGES = [
    ("pizza", "datasets/food-us-v0.1/images/train/pizza_001.jpg"),
    ("banana", "tests/assets/banana.jpg"),
    ("apple", "tests/assets/apple.jpg"),
    ("hot_dog", "datasets/food-us-v0.1/images/train/hot_dog_001.jpg"),
    ("salmon", "datasets/food-bench-v1/salmon/img_0001.jpg"),
    ("nachos", "datasets/food-bench-v1/nachos/img_0001.jpg"),
    ("lasagna", "datasets/food-bench-v1/lasagna/img_0001.jpg"),
    ("mac", "datasets/food-bench-v1/mac_and_cheese/img_0001.jpg"),
    ("mf_003", "datasets/multi-food/images/mf_003.jpg"),
    ("fries", "datasets/food-us-v0.1/images/train/french_fries_000.jpg"),
]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0

ok = fail = hangs = 0
lat = []
first_fail = None
for i in range(1, N + 1):
    name, rel = IMAGES[(i - 1) % len(IMAGES)]
    path = BASE / rel
    t0 = time.perf_counter()
    try:
        with open(path, "rb") as fh:
            resp = requests.post(API, files={"image": (path.name, fh, "image/jpeg")}, timeout=TIMEOUT)
        total = time.perf_counter() - t0
        lat.append(total)
        if resp.status_code == 200:
            ok += 1
            status = f"OK foods={len(resp.json().get('foods', []))}"
        else:
            fail += 1
            status = f"HTTP {resp.status_code}"
            if first_fail is None:
                first_fail = i
        print(f"[{i:>3}] {name:<10} {total*1000:>8.0f} ms  {status}", flush=True)
    except requests.exceptions.Timeout:
        hangs += 1
        fail += 1
        print(f"[{i:>3}] {name:<10} HANG (timeout {TIMEOUT}s)", flush=True)
        if first_fail is None:
            first_fail = i
        lat.append(TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        fail += 1
        print(f"[{i:>3}] {name:<10} ERROR {type(exc).__name__}: {exc}", flush=True)
        if first_fail is None:
            first_fail = i
    if i % 25 == 0:
        memory_report(f"mem[{i}]")

lat.sort()
print(f"\n=== RESULTADO ===")
print(f"requests={N} ok={ok} fail={fail} hangs={hangs} primer_fallo={first_fail}")
if lat:
    n = len(lat)
    print(f"p50={lat[n//2]*1000:.0f}ms p95={lat[int(n*0.95)]*1000:.0f}ms p99={lat[int(n*0.99)]*1000:.0f}ms max={lat[-1]*1000:.0f}ms")
memory_report("final")