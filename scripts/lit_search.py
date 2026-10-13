"""Documented literature search (novelty audit).

Runs a fixed list of queries against the arXiv API and writes the raw hits to
paper/lit_search/<date>.json plus a de-duplicated CSV. Re-run monthly until submission and
diff the CSVs to spot new related work:

    .venv\\Scripts\\python.exe scripts/lit_search.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "lit_search"

QUERIES = [
    'abs:"adaptive computation time"',
    'abs:"halting" AND abs:"recurrent" AND abs:"reasoning"',
    'abs:"ponder" AND abs:"halting"',
    'abs:"early exit" AND abs:"recurrent"',
    'abs:"deep thinking" AND abs:"extrapolat"',
    'abs:"easy-to-hard" AND abs:"recurrent"',
    'abs:"looped transformer" AND abs:"length generalization"',
    'abs:"recurrent depth" AND abs:"test-time compute"',
    'abs:"latent reasoning" AND abs:"adaptive"',
    'abs:"tiny recursive model"',
    'abs:"hierarchical reasoning model"',
    'abs:"when to stop" AND abs:"reasoning"',
    'abs:"termination" AND abs:"chain-of-thought"',
    'abs:"overthinking" AND abs:"reasoning"',
    'abs:"neural GPU" OR ti:"neural GPUs learn algorithms"',
    'abs:"carry" AND abs:"length generalization" AND abs:"addition"',
    'abs:"implicit" AND abs:"equilibrium" AND abs:"extrapolation"',
    'abs:"path independence" AND abs:"equilibrium"',
    'abs:"halting" AND abs:"out-of-distribution"',
    'abs:"confidence" AND abs:"halting" AND abs:"iterative"',
]


def query_arxiv(q: str, max_results: int = 50) -> list[dict]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": 0, "max_results": max_results, "sortBy": "relevance"}
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    hits = []
    for e in root.findall("a:entry", ns):
        arxiv_id = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
        hits.append(
            {
                "arxiv_id": arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id,
                "title": " ".join(e.find("a:title", ns).text.split()),
                "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                "published": e.find("a:published", ns).text[:10],
                "summary": " ".join(e.find("a:summary", ns).text.split()),
            }
        )
    return hits


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    raw = {}
    for q in QUERIES:
        try:
            raw[q] = query_arxiv(q)
        except Exception as exc:  # network hiccups are logged, not fatal
            raw[q] = {"error": repr(exc)}
        print(f"{q!r}: {len(raw[q]) if isinstance(raw[q], list) else raw[q]}")
        time.sleep(3.1)  # arXiv API etiquette
    (OUT_DIR / f"{today}.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")

    seen: dict[str, dict] = {}
    for q, hits in raw.items():
        if not isinstance(hits, list):
            continue
        for h in hits:
            rec = seen.setdefault(h["arxiv_id"], {**h, "queries": []})
            rec["queries"].append(q)
    with (OUT_DIR / f"{today}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "published", "n_queries", "title", "first_author"])
        for rec in sorted(seen.values(), key=lambda r: (-len(r["queries"]), r["published"])):
            w.writerow(
                [rec["arxiv_id"], rec["published"], len(rec["queries"]), rec["title"],
                 rec["authors"][0] if rec["authors"] else ""]
            )
    print(f"{len(seen)} unique papers -> {OUT_DIR / (today + '.csv')}")


if __name__ == "__main__":
    main()
