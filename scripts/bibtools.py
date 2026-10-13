"""Shared helpers for fetch_bib.py and check_bib.py: arXiv / DBLP / Crossref queries and a small
BibTeX parser."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

try:  # use the OS certificate store (uv-managed Python on Windows does not see it otherwise)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

UA = {"User-Agent": "glwm2-bibtools/0.1 (research reproducibility script)"}
ATOM = "{http://www.w3.org/2005/Atom}"


def http_get(url: str, headers: dict | None = None, retries: int = 4, pause: float = 3.0) -> str:
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # network hiccups and 429/503 rate limits
            last = e
            time.sleep(pause * (k + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def surname(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[{}\\'\"`^~]", "", name).strip()
    if "," in name:
        last = name.split(",")[0]
    else:
        last = name.split()[-1] if name.split() else name
    return re.sub(r"[^a-z]", "", last.lower())


def arxiv_meta(ids: list[str]) -> dict[str, dict]:
    """Title, authors, year and primary category for each arXiv id (batched, 3 s between calls)."""
    out = {}
    for s in range(0, len(ids), 40):
        chunk = ids[s : s + 40]
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": ",".join(chunk), "max_results": len(chunk)})
        root = ET.fromstring(http_get(url, retries=8, pause=20.0))
        for e in root.findall(f"{ATOM}entry"):
            aid = e.find(f"{ATOM}id").text.rsplit("/abs/", 1)[-1]
            aid = re.sub(r"v\d+$", "", aid)
            prim = e.find("{http://arxiv.org/schemas/atom}primary_category")
            jref = e.find("{http://arxiv.org/schemas/atom}journal_ref")
            out[aid] = {
                "title": " ".join(e.find(f"{ATOM}title").text.split()),
                "authors": [a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")],
                "year": e.find(f"{ATOM}published").text[:4],
                "primary": prim.get("term") if prim is not None else "cs.LG",
                "journal_ref": " ".join(jref.text.split()) if jref is not None and jref.text else "",
            }
        time.sleep(3.0)
    return out


_DBLP_BLOCKED = False


def dblp_search(title: str) -> list[dict]:
    global _DBLP_BLOCKED
    if _DBLP_BLOCKED:
        return []
    url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode({"q": title, "format": "json", "h": 20})
    txt = http_get(url)
    time.sleep(1.5)
    try:
        hits = json.loads(txt)["result"]["hits"].get("hit", [])
    except json.JSONDecodeError:  # DBLP serves a bot-check page to scripts; stop asking
        _DBLP_BLOCKED = True
        print("DBLP returned a bot check; falling back to arXiv + Semantic Scholar", flush=True)
        return []
    return [h["info"] for h in hits]


S2_RETRIES = 6


def s2_meta(arxiv_id: str, cache: dict | None = None) -> dict:
    """Venue and publication type from the Semantic Scholar Graph API ({} if unavailable).
    Successful responses are stored in `cache` so reruns do not hit the rate limit again."""
    if cache is not None and arxiv_id in cache:
        return cache[arxiv_id]
    url = (f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
           "?fields=title,venue,year,publicationVenue,publicationTypes,journal")
    try:
        d = json.loads(http_get(url, retries=S2_RETRIES, pause=10.0))
    except Exception:
        return {}
    finally:
        time.sleep(3.5)
    if cache is not None:
        cache[arxiv_id] = d
    return d


def dblp_bibtex(key: str) -> str:
    txt = http_get(f"https://dblp.org/rec/{key}.bib?param=1")
    time.sleep(1.5)
    return txt


def crossref_meta(doi: str) -> dict:
    msg = json.loads(http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    time.sleep(1.0)
    return {
        "title": " ".join((msg.get("title") or [""])[0].split()),
        "authors": [f"{a.get('family', '')}, {a.get('given', '')}" for a in msg.get("author", [])],
        "year": str((msg.get("issued", {}).get("date-parts") or [[None]])[0][0]),
    }


def doi_bibtex(doi: str) -> str:
    txt = http_get("https://doi.org/" + doi, headers={"Accept": "application/x-bibtex; charset=utf-8"})
    time.sleep(1.0)
    return txt


ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)


def parse_bib(text: str) -> dict[str, dict]:
    """Minimal BibTeX parser: entry type, key, and top-level fields (brace- or quote-delimited)."""
    entries = {}
    for m in ENTRY_RE.finditer(text):
        typ, key = m.group(1).lower(), m.group(2)
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end() : i - 1]
        fields, j = {}, 0
        while True:
            fm = re.compile(r"\s*(\w+)\s*=\s*").match(body, j)
            if not fm:
                break
            name, j = fm.group(1).lower(), fm.end()
            if j < len(body) and body[j] == "{":
                d, k = 1, j + 1
                while k < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val, j = body[j + 1 : k - 1], k
            elif j < len(body) and body[j] == '"':
                k = body.index('"', j + 1)
                val, j = body[j + 1 : k], k + 1
            else:
                k = j
                while k < len(body) and body[k] not in ",\n":
                    k += 1
                val, j = body[j:k].strip(), k
            fields[name] = " ".join(val.split())
            cm = re.compile(r"\s*,").match(body, j)
            j = cm.end() if cm else j
        entries[key] = {"type": typ, **fields}
    return entries


def bib_authors(field: str) -> list[str]:
    return [a.strip() for a in re.split(r"\s+and\s+", field) if a.strip()]
