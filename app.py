import time
import pandas as pd
import streamlit as st

from src.retrieval import retrieve_candidates
from src.rerank_xenc_local import rerank_local

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧭",
    layout="wide"
)

# ---- Sidebar controls
st.sidebar.title("Settings")
topk_candidates = st.sidebar.slider("Candidate pool (k)", 50, 300, 120, 10)
topk_results = st.sidebar.slider("Results to show", 5, 15, 10, 1)
st.sidebar.caption("Tip: Larger candidate pool usually improves accuracy slightly.")

st.title("SHL Assessment Recommender")
st.write("Paste a hiring query or job description below. The system will suggest the most relevant SHL assessments.")

example = (
    "We are hiring a QA Engineer with Selenium automation, Java/JS, SQL, and manual testing skills. "
    "Assessment duration should be around 60 minutes."
)
query = st.text_area("Query / Job Description", placeholder=example, height=180)

col_run, col_clear = st.columns([1,1], gap="small")
with col_run:
    run = st.button("Recommend", type="primary")
with col_clear:
    clear = st.button("Clear")

if clear:
    st.experimental_rerun()

# ---- Results section
if run:
    if not query or not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        t0 = time.time()
        cands = retrieve_candidates(query.strip(), k=topk_candidates)
        t1 = time.time()
        results = rerank_local(query.strip(), cands, k=topk_results)
        t2 = time.time()

        st.subheader("Top Recommendations")

        if not results:
            st.info("No results found. Try rephrasing the query or increasing the candidate pool.")
        else:
            df = pd.DataFrame(results)
            # Reorder / format columns
            if "url" in df.columns:
                df["Assessment"] = df.apply(lambda r: f"[{r.get('name','')}]({r.get('url','')})", axis=1)
            cols = [c for c in ["Assessment","score","test_type"] if c in df.columns]
            df_view = df[cols].rename(columns={
                "score": "Match Score",
                "test_type": "Type"
            })

            st.dataframe(
                df_view,
                use_container_width=True,
                hide_index=True
            )

            # Download
            out_csv = pd.DataFrame([{"name": r["name"], "url": r["url"], "score": r["score"], "type": r.get("test_type","")} for r in results])
            st.download_button(
                "Download CSV",
                data=out_csv.to_csv(index=False).encode("utf-8"),
                file_name="recommendations.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Perf
        st.caption(
            f"Retrieved in **{(t1-t0):.2f}s**, reranked in **{(t2-t1):.2f}s**. "
            f"Total: **{(t2-t0):.2f}s**."
        )

# ---- Footer
st.markdown("---")
st.caption("Built with FAISS + hybrid retrieval and a cross-encoder reranker.")
