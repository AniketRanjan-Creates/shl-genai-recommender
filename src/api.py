from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from src.retrieval import retrieve_candidates
from src.rerank_xenc_local import rerank_local

app = FastAPI()

class RecRequest(BaseModel):
    query: str
    k: int = 10

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/recommend")
def recommend(req: RecRequest):
    
    cands = retrieve_candidates(req.query, k=120)

    items = rerank_local(req.query, cands, k=req.k)
    return {"query": req.query, "results": items}
