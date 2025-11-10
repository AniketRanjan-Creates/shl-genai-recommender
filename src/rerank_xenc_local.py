import os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils_match import extract_duration_req, page_duration_minutes, query_implies_P

# ---- Global tiny TF-IDF model (fits on catalog once; tiny memory) ----
_catalog = None
_vec = None
_cat_text = None

def _load_catalog():
    global _catalog, _vec, _cat_text
    if _catalog is None:
        df = pd.read_parquet("data/catalog.parquet")
        _catalog = df[["url", "name", "text", "test_type"]].copy()
        _cat_text = (_catalog["name"].fillna("") + " \n " + _catalog["text"].fillna("")).values
        _vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")
        _vec.fit(_cat_text)  # very fast on ~50 docs

def _keyword_boost(query: str, name: str, url: str) -> float:
    q = query.lower()
    score = 0.0
    kws = [
        ("selenium", 0.6), ("manual testing", 0.4), ("java", 0.5), ("javascript", 0.5),
        ("html", 0.2), ("css", 0.2), ("sql", 0.5), ("excel", 0.3), ("python", 0.5),
        ("sales", 0.4), ("marketing", 0.4), ("verbal", 0.3), ("numerical", 0.3), ("opq", 0.3),
    ]
    for k,w in kws:
        if k in q: score += w

    # gentle nudges for stubborn slugs
    if "automata" in url: score += 0.2
    if "verify" in url:   score += 0.15
    if "opq" in url:      score += 0.15
    if "entry-level" in url or "entry_level" in url: score += 0.1

    # role hints
    if any(x in q for x in ["qa", "quality", "tester", "testing"]):
        if "selenium" in url or "manual" in url: score += 0.5

    return score

def _duration_boost(query: str, page_text: str) -> float:
    want = extract_duration_req(query)           # returns int minutes or None
    have = page_duration_minutes(page_text)      # best-effort scrape from text
    if not want or not have: return 0.0
    if have <= want: return 0.4                  # inside requested time
    if have <= want + 10: return 0.2             # slightly over
    return 0.0

def rerank_local(query: str, candidates: pd.DataFrame, k: int = 10):
    """
    Lite reranker: TF-IDF cosine over (name+text) + keyword & duration boosts.
    Avoids any heavy models so it runs on Render free (512MB).
    """
    _load_catalog()
    if candidates.empty:
        return []

    # join candidate meta from catalog
    df = candidates.merge(_catalog, on="url", how="left", suffixes=("", "_cat"))
    texts = (df["name"].fillna("") + " \n " + df["text"].fillna("")).values

    # TF-IDF cosine
    Q = _vec.transform([query])
    D = _vec.transform(texts)
    sims = cosine_similarity(Q, D).ravel()

    # boosts
    boosts = []
    qlower = query.lower()
    for n,u,t in zip(df["name"].fillna(""), df["url"], texts):
        boosts.append(_keyword_boost(qlower, n, u) + _duration_boost(qlower, t))
    boosts = np.array(boosts)

    # final score
    score = sims + 0.15 * boosts

    picked = df.assign(score=score).drop_duplicates(subset=["url"]).sort_values("score", ascending=False).head(k)
    out = []
    for _,r in picked.iterrows():
        out.append({
            "name": r.get("name") or "",
            "url": r.get("url"),
            "score": float(r.get("score", 0.0)),
            "test_type": (r.get("test_type") or "")[:1] or ""
        })
    return out
