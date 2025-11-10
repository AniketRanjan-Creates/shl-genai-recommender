import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from src.utils_match import extract_duration_req, page_duration_minutes, query_implies_P

# lazy-load the cross-encoder once
_CE = None
def _ce():
    global _CE
    if _CE is None:
        _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CE

def _duration_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    need = extract_duration_req(q)
    if not need or df.empty:
        return df
    out = df.copy()
    out["dur_min"] = [page_duration_minutes(str(x)) for x in out.get("description", "").astype(str)]
    keep = out[(out["dur_min"].isna()) | (out["dur_min"] <= need)]
    return keep if len(keep) else df

def _balance_after_rank(df: pd.DataFrame, q: str, k: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["test_type"] = df["test_type"].fillna("").replace({"": "K"})
    k = max(5, min(10, k))
    if not query_implies_P(q):
        return df.head(k)
    ks = df[df.test_type == "K"]
    ps = df[df.test_type == "P"]
    if len(ps) == 0 or len(ks) == 0:
        return df.head(k)
    target_p = max(2, k // 3)
    target_k = k - target_p
    take = pd.concat([ks.head(target_k), ps.head(target_p)], axis=0)
    if len(take) < k:
        rest = df[~df.index.isin(take.index)].head(k - len(take))
        take = pd.concat([take, rest], axis=0)
    return take.head(k)

def rerank_local(query: str, candidates: pd.DataFrame, k: int = 10):
    # guard: empty candidates
    if candidates is None or len(candidates) == 0:
        return []

    # duration pre-filter
    cands = _duration_filter(candidates.copy(), query)
    if cands.empty:
        cands = candidates.copy()

    # build pairs and score
    texts = cands.get("text", "").astype(str).tolist()
    pairs = [(query, t) for t in texts]
    if len(pairs) == 0:
        return []

    scores = _ce().predict(pairs)
    cands = cands.assign(xenc_score=scores).sort_values("xenc_score", ascending=False)

    # pick final set with knowledge/personality balance
    picked = _balance_after_rank(cands, query, k)
    if picked.empty:
        picked = cands.head(k)

    # deduplicate by URL and build output
    picked = picked.drop_duplicates(subset=["url"], keep="first")
    out = picked[["name", "url", "test_type"]].copy()
    out["score"] = picked["xenc_score"].round(4)
    out["reason"] = ""
    return out.to_dict("records")
