"""Cliente F46: ejecuta food-us completo contra el API real en shadow.

Escribe map.txt (analysis_id|filename). La telemetría (reranked1) se lee
después del stderr del server. Sin dependencias externas (urllib).
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
PORT = sys.argv[1] if len(sys.argv) > 1 else "8014"
DIR = BASE / "datasets/food-us-v0.1/images/train"
OUT = BASE / "benchmarks/f46/map.txt"


def multipart(analysis_id: str, image_bytes: bytes, filename: str) -> tuple[bytes, str]:
    boundary = "----f46" + uuid.uuid4().hex
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="analysis_id"\r\n\r\n{analysis_id}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode() + image_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fmap:
        for img in sorted(DIR.glob("*.jpg")):
            aid = str(uuid.uuid4())
            fmap.write(f"{aid}|{img.name}\n")
            body, ctype = multipart(aid, img.read_bytes(), img.name)
            req = urllib.request.Request(
                f"http://localhost:{PORT}/analyze", data=body, method="POST",
                headers={"Content-Type": ctype},
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                print(f"{img.name}: HTTP {exc.code}: {exc.read().decode()[:200]}", flush=True)
                continue
            assert data["status"] == "completed", f"status != completed: {data}"
            top = data["foods"][0]["name"] if data["foods"] else "none"
            print(f"{img.name}: {top} {time.perf_counter()-t0:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()