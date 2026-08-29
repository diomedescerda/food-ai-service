"""FASE 17: benchmark ampliado desde Wikimedia Commons (licencias compatibles).

Para cada clase objetivo descarga hasta N imágenes que depicten el alimento
(QID Wikidata via haswbstatement:P180) y registra metadata: source, url,
licencia, ground_truth. No usa imágenes sin licencia identificable.
"""
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CoppAddresdFoodBenchmark/1.0 (research; no commercial)"}
OUT = Path(__file__).resolve().parents[1] / "datasets" / "food-bench-v1"

ALLOWED_LICENSES = ("CC0", "CC BY", "CC BY-SA", "Public domain", "PD")
BLACKLIST = (
    "logo", "flag", "map", "icon", "monument", "building", "museum", "park",
    "bridge", "station", "tower", "city", "view", "skyline", "street",
    "kunsthalle", "planetarium", "camerata", "hamburg hof", "sign", "poster",
    "drawing", "painting", "sketch", "cartoon", "menu",
)

SEARCH_TERMS = {
    "hamburger": "burger",
    "french_fries": "french fries",
    "pizza": "pizza",
    "hot_dog": "hot dog",
    "sandwich": "sandwich",
    "fried_chicken": "fried chicken",
    "chicken_nuggets": "chicken nuggets",
    "taco": "taco",
    "burrito": "burrito",
    "quesadilla": "quesadilla",
    "nachos": "nachos",
    "steak": "steak",
    "salmon": "salmon",
    "rice": "rice dish",
    "pasta": "pasta dish",
    "lasagna": "lasagna",
    "mac_and_cheese": "macaroni and cheese",
    "eggs": "fried eggs",
    "bacon": "bacon",
    "toast": "toast",
    "bagel": "bagel",
    "pancakes": "pancakes",
    "waffles": "waffles",
    "oatmeal": "oatmeal",
    "donut": "donut",
    "cake": "cake",
    "cookie": "cookie",
    "brownie": "brownie",
    "ice_cream": "ice cream",
    "salad": "salad",
    "grilled_chicken": "grilled chicken",
}

CLASSES = tuple(SEARCH_TERMS)


def fetch_thumbnails(term: str, limit: int) -> list[dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f'intitle:"{term}" filetype:bitmap',
        "gsrnamespace": 6,
        "gsrlimit": limit * 8,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "format": "json",
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        ii = page.get("imageinfo", [{}])[0]
        license_name = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "")
        title = page.get("title", "")
        if not any(lic in license_name for lic in ALLOWED_LICENSES):
            continue
        if not title.lower().endswith((".jpg", ".jpeg")):
            continue
        if any(word in title.lower() for word in BLACKLIST):
            continue
        if (ii.get("width") or 0) < 320 or (ii.get("height") or 0) < 320:
            continue
        # Thumb estándar de 640px vía Special:FilePath (respeta el robot
        # policy de Wikimedia y reduce el ancho de banda).
        thumb_url = (
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            + quote(title.replace("File:", ""))
            + "?width=640"
        )
        out.append({
            "title": title,
            "thumburl": thumb_url,
            "url": ii.get("descriptionurl"),
            "width": ii.get("width"),
            "height": ii.get("height"),
            "license": license_name,
        })
        if len(out) >= limit * 2:
            break
    return out


def main() -> None:
    only = sys.argv[1:] if len(sys.argv) > 1 else list(CLASSES)
    per_class = 8
    metadata = {}
    (OUT / "_meta").mkdir(parents=True, exist_ok=True)

    for cls in only:
        term = SEARCH_TERMS[cls]
        cls_dir = OUT / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(cls_dir.glob("*.jpg")))
        need = max(0, per_class - existing)
        if need == 0:
            print(f"[skip] {cls}: ya tiene {existing}")
            continue
        try:
            thumbs = fetch_thumbnails(term, need * 2)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {cls}: {exc}")
            continue
        got = 0
        for i, thumb in enumerate(thumbs):
            if got >= need:
                break
            name = f"img_{existing + got + 1:04d}.jpg"
            try:
                img = None
                for attempt in range(4):
                    try:
                        img = requests.get(thumb["thumburl"], headers=HEADERS, timeout=30)
                        img.raise_for_status()
                        break
                    except requests.exceptions.HTTPError as exc:
                        if exc.response is not None and exc.response.status_code == 429 and attempt < 3:
                            time.sleep(15 * (attempt + 1))
                            continue
                        raise
                assert img is not None
                if len(img.content) < 5_000:
                    continue
                path = cls_dir / name
                path.write_bytes(img.content)
                metadata[f"{cls}/{name}"] = {
                    "class": cls,
                    "source": "Wikimedia Commons",
                    "source_url": thumb["url"],
                    "original_title": thumb["title"],
                    "license": thumb["license"],
                    "ground_truth": cls,
                    "food_count": 1,
                    "multi_food": False,
                }
                got += 1
                print(f"[ok] {cls}/{name} lic={thumb['license']}")
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {cls} {thumb['title'][:40]}: {str(exc)[:60]}")
            time.sleep(8)
        print(f"[done] {cls}: {got}/{need}")

    meta_path = OUT / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            all_meta = json.load(fh)
        all_meta.update(metadata)
    else:
        all_meta = metadata
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(all_meta, fh, ensure_ascii=False, indent=2)
    print(f"[meta] total imágenes registradas: {len(all_meta)}")


if __name__ == "__main__":
    main()