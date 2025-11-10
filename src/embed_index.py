import os, numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

os.makedirs("index", exist_ok=True)
df = pd.read_parquet("data/catalog.parquet")
df["text"] = df["text"].fillna("").astype(str)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
X = X.astype("float32")
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
faiss.write_index(index, "index/faiss.index")
np.save("index/embeddings.npy", X)
df.to_parquet("data/catalog.parquet", index=False)
print("indexed", len(df))
