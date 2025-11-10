# SHL Assessment Recommender (Generative AI Assignment)

A retrieval + reranking system that recommends SHL assessments based on hiring queries or job descriptions.

## 🧠 Overview

This project builds a **Retrieval-Augmented Generation (RAG)** pipeline over SHL's assessment catalog.

It:
- Indexes the SHL product catalog using FAISS.
- Retrieves top-k candidates using text embeddings.
- Optionally reranks results using a cross-encoder (local version for cost-free evaluation).
- Provides a **Streamlit UI** for easy querying and CSV download.

---

## ⚙️ Setup

```bash
git clone <your_repo_url>
cd shl_rec
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
