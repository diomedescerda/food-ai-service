"""Resumen A/B/C de clasificadores."""
import json

r = json.load(open(r"benchmark_classifier_ab_results.json", encoding="utf-8"))
for m in ("clip", "dinov3", "beit"):
    print(f"=== {m}")
    for ds in ("foodus", "v1"):
        d = r[m].get(ds, {})
        n = d.get("total", 0)
        if n:
            print(f"  {ds}: top1={d['top1']/n*100:.1f}% top3={d['top3']/n*100:.1f}% "
                  f"top5={d['top5']/n*100:.1f}% unknown={d['unknown']} ({d['unknown']/n*100:.1f}%) "
                  f"lat={d['lat_ms_mean']:.0f}ms")
    print(f"  startup={r[m].get('startup_s')}s")