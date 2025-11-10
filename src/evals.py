import pandas as pd
from src.retrieval import retrieve_candidates
from src.rerank_xenc_local import rerank_local
from src.utils_match import canon_url

# replace recall_at_k with this
def recall_at_k(pred, gold, k=10):
    P = set(canon_url(u) for u in pred[:k] if isinstance(u, str))
    G = set(canon_url(u) for u in gold if isinstance(u, str))
    return 0.0 if not G else len(P & G) / len(G)



def run_eval(label_path="data/labeled_train.csv", k=10):
    """Run mean recall@k over all labeled queries."""
    gold = pd.read_csv(label_path)
    gold["Assessment_url"] = gold["Assessment_url"].astype(str)
    queries = gold["Query"].dropna().unique().tolist()
    scores = []
    for q in queries:
        gurls = gold[gold.Query == q]["Assessment_url"].tolist()

        cands = retrieve_candidates(q, k=120)

        preds = rerank_local(q, cands, k=k)
        pred_urls = [p["url"] for p in preds]
        r = recall_at_k(pred_urls, gurls, k=k)
        scores.append(r)
        print(f"R@{k}: {r:.3f} | {q[:70]}...")
    mr = sum(scores) / len(scores)
    print(f"\nMean Recall@{k}: {mr:.3f}")
    return mr

if __name__ == "__main__":
    run_eval()
