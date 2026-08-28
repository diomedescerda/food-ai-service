"""Build del dataset US Food MVP (food-us-v0.1).

Fuente: Wikimedia Commons (API MediaWiki). Por cada clase descarga N imágenes
(~640 px), filtra licencias (CC0/CC BY/CC BY-SA/PD), genera metadata con
trazabilidad completa, deduplica (MD5 + pHash) y reparte en train/val/test
(70/20/10, seed fija).

Uso:
    python scripts/build_dataset.py --target 25 --classes pizza,hamburger

Los datasets pesados NO van a git (datasets/*/images/ está gitignoreado).
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "food-us-v0.1"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FoodAI-DatasetBuilder/0.1 (dev; contact: local)"

ALLOWED_LICENSES = ("cc0", "cc by", "cc by-sa", "public domain")
EXCLUDED_LICENSES = ("cc by-nc", "cc by-nd", "gfdl")

# Categorías de Commons por clase (se prueban en orden hasta completar target).
CLASS_CATEGORIES: dict[str, list[str]] = {
    "pizza": ["Pizza"],
    "hamburger": ["Hamburgers"],
    "french_fries": ["French fries", "Fried potatoes"],
    "hot_dog": ["Hot dogs"],
    "sandwich": ["Sandwiches"],
    "fried_chicken": ["Fried chicken"],
    "salad": ["Salads"],
    "pasta": ["Pasta dishes", "Spaghetti dishes"],
    "steak": ["Steaks"],
    "rice": ["Rice dishes"],
    "pancakes": ["Pancakes"],
    "tacos": ["Tacos"],
}

CLASS_CATEGORY_TAG = {
    "pizza": "fast_food" if False else "restaurant",
    "hamburger": "fast_food",
    "french_fries": "fast_food",
    "hot_dog": "fast_food",
    "sandwich": "restaurant",
    "fried_chicken": "fast_food",
    "salad": "restaurant",
    "pasta": "restaurant",
    "steak": "restaurant",
    "rice": "restaurant",
    "pancakes": "breakfast",
    "tacos": "restaurant",
}


def api_request(params: dict) -> dict:
    params["format"] = "json"
    url = API_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 30 * (attempt + 1)
                print(f"[warn] 429 Too Many Requests — esperando {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("rate limit agotado")


def category_files(category: str, limit: int = 50) -> list[str]:
    """Nombres de archivos de una categoría (paginado, hasta `limit`)."""
    files: list[str] = []
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": "50",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": "640",
        "redirects": "1",
    }
    while len(files) < limit:
        data = api_request(params)
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            title = page.get("title", "")
            if not title.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            files.append(title)
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
    return files


def image_license(page: dict) -> tuple[str, str] | None:
    """(license_name, license_url) si la licencia está permitida; None si no."""
    info = (page.get("imageinfo") or [{}])[0]
    ext = info.get("extmetadata") or {}
    name = (ext.get("LicenseShortName") or {}).get("value", "").strip()
    url = (ext.get("LicenseUrl") or {}).get("value", "").strip()
    lower = name.lower()
    if any(bad in lower for bad in EXCLUDED_LICENSES):
        return None
    if any(ok in lower for ok in ALLOWED_LICENSES):
        return (name, url)
    return None


def query_titles_license(titles: list[str], candidates: list) -> int:
    """Consulta licencia/MIME/thumb en lotes de 10 (robusto: lotes grandes
    truncan la respuesta; además respeta el rate limit). Añade a `candidates`
    y devuelve cuántos encontró."""
    found = 0
    for start in range(0, len(titles), 10):
        batch = titles[start:start + 10]
        try:
            data = api_request({
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime",
                "iiurlwidth": "640",
            })
        except Exception as exc:
            print(f"[warn] lote de títulos: {exc}")
            continue
        for page in (data.get("query") or {}).get("pages", {}).values():
            lic = image_license(page)
            if lic is None:
                continue
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            thumb = info.get("thumburl") or info.get("url")
            if not thumb:
                continue
            candidates.append((page["title"], lic[0], lic[1], thumb))
            found += 1
        time.sleep(0.8)
    return found


def download(url: str, dest: Path) -> bool:
    # UA de navegador: el throttling de uploads responde mejor a UAs reales.
    browser_ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": browser_ua})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            dest.write_bytes(data)
            with Image.open(dest) as img:
                if img.width < 200:
                    dest.unlink(missing_ok=True)
                    return False
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Sin retry bloqueante: las fallidas se reintentan al final
                # del loop de la clase (ritmo constante evita más 429).
                dest.unlink(missing_ok=True)
                return False
            dest.unlink(missing_ok=True)
            return False
        except Exception:
            dest.unlink(missing_ok=True)
            return False
    dest.unlink(missing_ok=True)
    return False


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def phash(path: Path) -> int:
    """Average hash de 64 bits (8x8)."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((8, 8))
        pixels = list(gray.getdata())
        mean = sum(pixels) / len(pixels)
    bits = 0
    for i, pixel in enumerate(pixels):
        if pixel >= mean:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=25)
    parser.add_argument("--classes", default=None, help="coma-separado; default: todas")
    parser.add_argument("--min-width", type=int, default=200)
    args = parser.parse_args()

    classes = args.classes.split(",") if args.classes else list(CLASS_CATEGORIES)
    for cls in classes:
        if cls not in CLASS_CATEGORIES:
            print(f"[skip] clase desconocida: {cls}")
            classes.remove(cls)

    images_dir = DATASET_DIR / "images"
    metadata_dir = DATASET_DIR / "metadata"
    for split in ("train", "val", "test"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    sources: dict[str, dict] = {}
    license_counts: dict[str, int] = {}

    # Resume: cargar metadata previa si existe (corridas interrumpidas).
    prev_images_csv = metadata_dir / "images.csv"
    if prev_images_csv.exists():
        with prev_images_csv.open(encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
        print(f"[resume] metadata previa: {len(rows)} filas cargadas")

    for cls in classes:
        target = args.target
        per_category = max(target * 3, 90)
        titles: list[str] = []

        # 1) Categorías candidatas (hasta completar títulos suficientes)
        for category in CLASS_CATEGORIES[cls]:
            found = category_files(category, limit=per_category)
            print(f"[info] {cls}: categoria '{category}' -> {len(found)} archivos")
            titles.extend(found)
            if len(titles) >= per_category:
                break

        # 2) Fallback: búsqueda por texto si las categorías no dieron suficiente
        if len(titles) < per_category:
            try:
                data = api_request({
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"filetype:bitmap {cls}",
                    "gsrnamespace": "6",
                    "gsrlimit": "50",
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime",
                })
                extra = []
                for page in (data.get("query") or {}).get("pages", {}).values():
                    title = page.get("title", "")
                    if title.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        extra.append(title)
                titles.extend(extra)
                print(f"[info] {cls}: fallback busqueda -> {len(extra)} archivos")
            except Exception as exc:
                print(f"[warn] fallback busqueda {cls}: {exc}")

        # 3) Filtro de licencia + thumbs en lotes de 10
        candidates: list[tuple[str, str, str, str]] = []
        query_titles_license(titles, candidates)
        print(f"[info] {cls}: {len(candidates)} candidatos con licencia válida")

        # Descargas SECUENCIALES con RESUME: si una corrida anterior quedó a
        # medias (el proceso puede morir), las imágenes ya descargadas se
        # cuentan y el target continúa desde ahí.
        existing = list((images_dir / "train").glob(f"{cls}_*.jpg"))
        downloaded = len(existing)
        if downloaded > 0:
            print(f"[resume] {cls}: {downloaded} imágenes ya descargadas")

        seen_titles: set[str] = set()
        for title, license_name, license_url, thumb_url in candidates:
            if downloaded >= target:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)
            if license_name not in license_counts:
                license_counts[license_name] = 0

            image_id = f"{cls}_{downloaded:03d}"
            dest = images_dir / "train" / f"{image_id}.jpg"  # split provisional
            if download(thumb_url, dest):
                license_counts[license_name] += 1
                rows.append({
                    "image_id": image_id,
                    "filename": f"{image_id}.jpg",
                    "class": cls,
                    "split": "train",  # se reasigna después
                    "source": "wikimedia_commons",
                    "source_url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                    "license": license_name,
                    "license_url": license_url,
                    "category": CLASS_CATEGORY_TAG.get(cls, "food"),
                    "downloaded_at": "2026-08-28",
                })
                sources["wikimedia_commons"] = {
                    "source": "wikimedia_commons",
                    "license": "por imagen (CC0/CC BY/CC BY-SA/PD)",
                    "attribution": "URL de la imagen en metadata",
                    "url": "https://commons.wikimedia.org",
                }
                downloaded += 1
                print(f"[ok] {cls}: {downloaded}/{target} ({Path(title).name[:40]}...)")
            time.sleep(3.0)

        if downloaded < target:
            print(f"[GAP] {cls}: {downloaded}/{target} - DATASET_GAP")

    if not rows:
        print("No se descargó ninguna imagen.")
        return

    # Deduplicación (MD5 + pHash) sobre el conjunto completo
    print("[dedup] ejecutando MD5 + pHash...")
    unique: list[dict] = []
    hashes: list[int] = []
    md5s: set[str] = set()
    removed = 0
    for row in rows:
        path = images_dir / "train" / row["filename"]
        digest = md5_of(path)
        if digest in md5s:
            removed += 1
            path.unlink(missing_ok=True)
            continue
        h = phash(path)
        if any(hamming(h, other) <= 8 for other in hashes):
            removed += 1
            path.unlink(missing_ok=True)
            continue
        md5s.add(digest)
        hashes.append(h)
        unique.append(row)

    # Splits 70/20/10 con seed fija
    import random

    random.seed(42)
    by_class: dict[str, list[dict]] = {}
    for row in unique:
        by_class.setdefault(row["class"], []).append(row)

    for cls, cls_rows in by_class.items():
        random.shuffle(cls_rows)
        n = len(cls_rows)
        n_train = max(1, round(n * 0.7))
        n_val = max(1, round(n * 0.2))
        for i, row in enumerate(cls_rows):
            row["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    # Mover archivos a su split final
    for row in unique:
        src = images_dir / "train" / row["filename"]
        dest = images_dir / row["split"] / row["filename"]
        src.replace(dest)

    # Metadata
    images_csv = metadata_dir / "images.csv"
    with images_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(unique[0].keys()))
        writer.writeheader()
        writer.writerows(unique)

    sources_csv = metadata_dir / "sources.csv"
    with sources_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "license", "attribution", "url"])
        writer.writeheader()
        writer.writerows(sources.values())

    classes_csv = metadata_dir / "classes.csv"
    with classes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "priority", "category", "target_count"])
        writer.writeheader()
        for cls in classes:
            writer.writerow({
                "class": cls,
                "priority": "HIGH" if cls in ("pizza", "hamburger", "french_fries", "hot_dog",
                                              "sandwich", "fried_chicken", "salad", "pasta",
                                              "steak", "rice") else "MEDIUM",
                "category": CLASS_CATEGORY_TAG.get(cls, "food"),
                "target_count": target,
            })

    (DATASET_DIR / "splits.json").write_text(
        json.dumps({"seed": 42, "train": 0.7, "val": 0.2, "test": 0.1}, indent=2),
        encoding="utf-8",
    )

    # dataset.yaml
    names = "\n".join(f"  {i}: {cls}" for i, cls in enumerate(classes))
    (DATASET_DIR / "dataset.yaml").write_text(
        f"""# Food-US dataset v0.1 — MVP (metadata; anotaciones en FASE 11)
# Clases CONGELADAS: no reordenar ids sin nueva versión.
path: datasets/food-us-v0.1
train: images/train
val: images/val
test: images/test
names:
{names}
""",
        encoding="utf-8",
    )

    stats = {
        "total": len(unique),
        "removed_duplicates": removed,
        "per_class": {cls: len(rows) for cls, rows in by_class.items()},
        "splits": {
            split: sum(1 for r in unique if r["split"] == split)
            for split in ("train", "val", "test")
        },
        "licenses": license_counts,
    }
    (DATASET_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n=== DATASET food-us-v0.1 ===")
    print(f"Imágenes: {stats['total']} (dedup eliminó {removed})")
    print(f"Splits: {stats['splits']}")
    print(f"Por clase: {stats['per_class']}")
    print(f"Licencias: {stats['licenses']}")


if __name__ == "__main__":
    main()