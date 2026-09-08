"""F56: rollout gradual — split de tráfico por etapas (0/10/25/50/100%).

Usa DOS instancias: server_new (FOOD_AI_RETRIEVAL_ENABLED=true, puerto A)
y server_legacy (flag off, puerto B). El cliente enruta X% al new y el resto
al legacy, mide accuracy (GT del filename), fallbacks, errores y latencia.

Uso: python scripts/f56_rollout.py <port_new> <port_legacy>
"""
import json
import re
import sys
import time
import uuid
from pathlib import Path

import urllib.error
import urllib.request

BASE = Path(__file__).resolve().parents[1]
DIR = BASE / "datasets/food-us-v0.1/images/train"

STAGES = (0.0, 0.10, 0.25, 0.50, 1.0)


def multipart(analysis_id: str, image_bytes: bytes, filename: str) -> tuple[bytes, str]:
    boundary = "----f56" + uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="analysis_id"\r\n\r\n{analysis_id}\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode() + image_bytes + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def send(port: int, image_bytes: bytes, filename: str) -> tuple[str, float, bool]:
    aid = str(uuid.uuid4())
    body, ctype = multipart(aid, image_bytes, filename)
    req = urllib.request.Request(
        f"http://localhost:{port}/analyze", data=body, method="POST",
        headers={"Content-Type": ctype},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        top = data["foods"][0]["name"] if data.get("foods") else "none"
        return top, time.perf_counter() - t0, False
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{exc}", time.perf_counter() - t0, True


def main() -> None:
    port_new = int(sys.argv[1])
    port_legacy = int(sys.argv[2])
    images = sorted(DIR.glob("*.jpg"))
    print("stage,new_pct,requests,new_used,legacy_used,new_correct,legacy_correct,"
          "new_incorrect,legacy_incorrect,fallbacks,errors,p50_ms,p95_ms,observations")
    for pct in STAGES:
        new_used = legacy_used = 0
        new_correct = legacy_correct = new_incorrect = legacy_incorrect = 0
        fallbacks = errors = 0
        lats = []
        obs = []
        for img in images:
            gt = img.stem.rsplit("_", 1)[0].replace("_", " ")
            use_new = (hash(img.name) % 100) / 100.0 < pct
            port = port_new if use_new else port_legacy
            food, lat, err = send(port, img.read_bytes(), img.name)
            lats.append(lat * 1000)
            if err:
                errors += 1
                continue
            if use_new:
                new_used += 1
                if food == gt:
                    new_correct += 1
                else:
                    new_incorrect += 1
                    if food == "none":
                        fallbacks += 1
            else:
                legacy_used += 1
                if food == gt:
                    legacy_correct += 1
                else:
                    legacy_incorrect += 1
            if img.stem.startswith(("pizza", "hamburger", "fries", "hot_dog", "sandwich")):
                obs.append(f"{img.stem.split('_')[0]}:{food}")
        lats.sort()
        p50 = lats[len(lats) // 2]
        p95 = lats[int(len(lats) * 0.95)]
        print(f"{int(pct*100)}%,{pct:.2f},{len(images)},{new_used},{legacy_used},"
              f"{new_correct},{legacy_correct},{new_incorrect},{legacy_incorrect},"
              f"{fallbacks},{errors},{p50:.0f},{p95:.0f},{';'.join(obs[:5])}", flush=True)
    print("ROLLOUT DONE", flush=True)


if __name__ == "__main__":
    main()