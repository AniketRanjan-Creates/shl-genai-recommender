import pandas as pd
from src.retrieval import retrieve_candidates
from src.rerank_xenc_local import rerank_local
from src.utils_match import canon_url

gold = pd.read_csv("data/labeled_train.csv")
gold["Assessment_url"] = gold["Assessment_url"].astype(str)

def inspect(query):
    G = [canon_url(u) for u in gold[gold.Query==query]["Assessment_url"].tolist()]
    print("GOLD:")
    for u in G: print(" -", u)
    cands = retrieve_candidates(query, k=120)
    preds = rerank_local(query, cands, k=10)
    print("\nPRED:")
    for i,p in enumerate(preds,1):
        print(f"{i:02d}.", canon_url(p["url"]), "|", p["name"])

if __name__ == "__main__":
    q = "Based on the JD below recommend me assessment for the Consultant position in my organizations. The assessment should not"
    inspect(q)
