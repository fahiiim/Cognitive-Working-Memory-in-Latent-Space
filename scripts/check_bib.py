"""Verify paper/references.bib against the arXiv API and Crossref, and the citations in paper/*.tex.

Fails (exit 1) if any entry's title or first-author surname disagrees with the source record, if an
entry has neither an arXiv id nor a DOI, or if a \\cite key in the paper is missing from the bib.

    .venv\\Scripts\\python.exe scripts\\check_bib.py [--offline]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bibtools import arxiv_meta, bib_authors, crossref_meta, norm_title, parse_bib, surname  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "references.bib"
CITE_RE = re.compile(r"\\(?:cite|citet|citep|citealp|citeauthor|citeyear)\*?(?:\[[^\]]*\])*\{([^}]*)\}")


def cited_keys() -> set[str]:
    keys = set()
    for tex in (ROOT / "paper").rglob("*.tex"):
        for m in CITE_RE.finditer(tex.read_text(encoding="utf-8")):
            keys |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="only check cite keys and identifiers")
    args = ap.parse_args()
    bib = parse_bib(BIB.read_text(encoding="utf-8"))
    errors = []
    cited = cited_keys()
    for k in sorted(cited - set(bib)):
        errors.append(f"cited but not in references.bib: {k}")
    unused = sorted(set(bib) - cited)
    for k, e in bib.items():
        if "eprint" not in e and "doi" not in e:
            errors.append(f"{k}: no eprint or doi to verify against")
    if not args.offline:
        ax = {k: e["eprint"] for k, e in bib.items() if "eprint" in e}
        meta = arxiv_meta(sorted(set(ax.values())))
        for k, aid in ax.items():
            m = meta.get(aid)
            if m is None:
                errors.append(f"{k}: arXiv id {aid} not found")
                continue
            e = bib[k]
            if norm_title(e.get("title", "")) != norm_title(m["title"]):
                errors.append(f"{k}: title mismatch\n    bib:   {e.get('title')}\n    arXiv: {m['title']}")
            ba = bib_authors(e.get("author", ""))
            if not ba or surname(ba[0]) != surname(m["authors"][0]):
                errors.append(f"{k}: first author mismatch ({ba[:1]} vs {m['authors'][:1]})")
        for k, e in bib.items():
            if "eprint" in e or "doi" not in e:
                continue
            cm = crossref_meta(e["doi"])
            if norm_title(e.get("title", "")) != norm_title(cm["title"]):
                errors.append(f"{k}: title mismatch\n    bib:      {e.get('title')}\n    Crossref: {cm['title']}")
            ba = bib_authors(e.get("author", ""))
            if cm["authors"] and (not ba or surname(ba[0]) != surname(cm["authors"][0])):
                errors.append(f"{k}: first author mismatch ({ba[:1]} vs {cm['authors'][:1]})")
    print(f"{len(bib)} entries, {len(cited)} cited keys, {len(unused)} unused entries")
    if unused:
        print("  unused:", ", ".join(unused))
    for e in errors:
        print("ERROR", e)
    print("check_bib:", "FAILED" if errors else "OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
