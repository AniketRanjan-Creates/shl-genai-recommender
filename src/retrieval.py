import re
import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils_match import extract_duration_req, page_duration_minutes  # duration helpers

# ----------------------------
# Load data + dense index
# ----------------------------
df = pd.read_parquet("data/catalog.parquet")
for col in ["name","url","text","description","test_type"]:
    df[col] = df.get(col, "").fillna("").astype(str)

emb = np.load("index/embeddings.npy")
index = faiss.read_index("index/faiss.index")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Sparse vectorizer over title+url+text
# ----------------------------
corpus = (df["name"] + " " + df["url"] + " " + df["text"]).tolist()
_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=80000, lowercase=True)
_Xsparse = _vectorizer.fit_transform(corpus)

# ----------------------------
# Helpers
# ----------------------------
# Use NON-capturing groups (?:...) to avoid the pandas warning, and word boundaries.
# ----------------------------
# Helpers
# ----------------------------
# Non-capturing regex and word boundaries; these only activate if the QUERY contains them
SKILL_PATTERNS = [
    (re.compile(r"\bjava(?:script)?\b", re.I), 0.15),
    (re.compile(r"\bpython\b", re.I),           0.15),
    (re.compile(r"\bsql\b", re.I),              0.15),
    (re.compile(r"\bselenium\b", re.I),         0.18),  # QA-heavy
    (re.compile(r"\bexcel\b", re.I),            0.12),
    (re.compile(r"\bhtml\b", re.I),             0.10),
    (re.compile(r"\bcss\b", re.I),              0.10),
    (re.compile(r"\btableau\b", re.I),          0.10),
    (re.compile(r"\bopq\b|\bpersonality\b|\binterpersonal\b|\bcollaborat", re.I), 0.12),
    (re.compile(r"\bsales\b|\bmarketing\b", re.I), 0.08),
]

# Role-specific nudges: if query implies the role, add targeted boosts to titles/urls
ROLE_HINTS = [
    # QA Engineer – automation + web + SQL + 1h profiles
    (
        re.compile(r"\bqa\b|quality\s*assurance|test\s*engineer|selenium", re.I),
        [
            (r"automata\s*selenium|selenium\b",                             0.35),
            (r"\bmanual\s*testing\b",                                       0.25),
            (r"\bautomata\s*sql\b|\bsql\s*(?:server)?\b",                    0.18),
            (r"\bjava(?:script)?\b|\bhtml\b|\bcss\b",                        0.15),
            (r"professional\s*7(?:\.1)?\b|professional\s+solution",          0.10),
        ],
    ),
    # Consultant / I-O psychology – OPQ + Verify + Numerical + Professional
    (
        re.compile(r"\bconsultant\b|industrial|organizational|\bi/?o\b|psycholog", re.I),
        [
            (r"\bopq\b|occupational\s*personality",                          0.30),
            (r"\bverify\s*verbal\b|verbal\s*ability",                        0.20),
            (r"interactive\s*numerical|numerical\s*calculation",             0.20),
            (r"professional\s*7(?:\.1)?\b|professional\s+solution",          0.12),
        ],
    ),
]


def _keyword_boost(query: str, names: pd.Series, urls: pd.Series):
    q = (query or "").lower()
    w = np.zeros(len(names), dtype=np.float32)

    # 1) Base skill patterns – apply only if query mentions them
    for pat, bonus in SKILL_PATTERNS:
        if pat.search(q):
            hit = names.str.contains(pat, na=False) | urls.str.contains(pat, na=False)
            w[hit.values] += bonus

    # 2) Role hints – if query implies the role, add targeted boosts
    for qpat, boosts in ROLE_HINTS:
        if qpat.search(q):
            for bpat, bonus in boosts:
                p = re.compile(bpat, re.I)
                hit = names.str.contains(p, na=False) | urls.str.contains(p, na=False)
                w[hit.values] += bonus

                

    # 3) Tiny direct token overlap (title > url)
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    title_tok_overlap = names.str.lower().apply(lambda s: len(q_tokens & set(re.findall(r"[a-z0-9]+", s))))
    url_tok_overlap   = urls.str.lower().apply(lambda s: len(q_tokens & set(re.findall(r"[a-z0-9]+", s))))
    w += 0.02 * title_tok_overlap.to_numpy(dtype=np.float32)
    w += 0.01 * url_tok_overlap.to_numpy(dtype=np.float32)
    return w

def _duration_boost(query: str, descs: pd.Series):
    # if query specifies time, reward items within that budget; stronger push than before
    need = extract_duration_req(query)  # minutes or None
    if not need:
        return np.zeros(len(descs), dtype=np.float32)
    mins = descs.apply(lambda t: page_duration_minutes(str(t)) or 9999).to_numpy()
    w = np.zeros_like(mins, dtype=np.float32)
    # inside budget → big boost; slightly reward <= need+10 in case of off-by-few-minutes
    w[mins <= need] = 1.0
    w[(mins > need) & (mins <= need + 10)] = 0.4
    return w


# ----------------------------
# Dense + Sparse + Keyword + Duration combo
# ----------------------------
def _dense_scores(query: str, n: int = 300):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, n)
    return D[0], I[0]

def _sparse_scores(query: str):
    qv = _vectorizer.transform([query])
    s = (_Xsparse @ qv.T).toarray().ravel()
    return s

def retrieve_candidates(query: str, k: int = 120, w_dense: float = 0.50, w_sparse: float = 0.30, w_kw: float = 0.12, w_dur: float = 0.08):
    ...

    # dense
    D, I = _dense_scores(query, n=max(k, 300))
    # sparse
    s_sparse = _sparse_scores(query)

    # candidate pool = union of dense top-N and sparse top-N (bigger pool reduces R@0)
    top_dense  = set(I[:max(220, k)])
    top_sparse = set(np.argsort(s_sparse)[-max(220, k):])
    pool_idx = np.array(sorted(top_dense | top_sparse))

    # normalize dense & sparse on the pool
    d = D.copy()
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
    s = s_sparse
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)

    names = df.iloc[pool_idx]["name"]
    urls  = df.iloc[pool_idx]["url"]
    descs = df.iloc[pool_idx]["description"]

    kb   = _keyword_boost(query, names, urls)
    kb   = (kb - kb.min()) / (kb.max() - kb.min() + 1e-9)

    kdur = _duration_boost(query, descs)
    if kdur.max() > 0:
        kdur = kdur / (kdur.max() + 1e-9)

    
    combo = w_dense * d_norm[pool_idx] + w_sparse * s_norm[pool_idx] + w_kw * kb + w_dur * kdur

    order = np.argsort(-combo)[:k]
    chosen = pool_idx[order]

    out = df.iloc[chosen].copy()
    out["score"] = combo[order]
    out = out.drop_duplicates(subset=["url"], keep="first")
    return out

