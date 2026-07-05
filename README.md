# NLP Semantic Search System for Research Papers

A semantic search engine for ML research papers that goes beyond keyword matching — it understands meaning, retrieves the most relevant papers, summarizes their abstracts, and extracts key topics automatically.

## Overview

This project builds an end-to-end pipeline:

```
Query → Semantic Search (FAISS) → Top-k Relevant Papers → Summarization → Keyword Extraction
```

Given a natural language query like *"deep learning for medical image analysis"*, the system finds the most semantically similar papers from a corpus of 15,000 ML arXiv papers, generates concise summaries of their abstracts, and extracts key topics/keywords.

## Features

- **Semantic Search** — Uses sentence embeddings (not just keyword matching) to understand query intent.
- **Fast Vector Retrieval** — FAISS-based similarity search over paper embeddings.
- **Automatic Summarization** — Condenses paper abstracts into short, readable summaries.
- **Keyword Extraction** — Surfaces key terms/topics from each paper.
- **Persistent Caching** — Embeddings and the FAISS index are saved to disk so they don't need to be recomputed on every run.

## Dataset

- **Source**: [`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (Hugging Face)
- **Size used**: 15,000 papers
- **Fields used**: `title` + `abstract`, combined into a single `paper_text` field

## Tech Stack

| Component | Tool / Model |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| Vector search | `faiss` (`IndexFlatIP`, cosine similarity via L2-normalized vectors) |
| Summarization | `transformers` (`facebook/bart-large-cnn`) |
| Keyword extraction | `KeyBERT` |
| Data handling | `pandas` |

## Pipeline

### 1. Data Preparation
Load the dataset, keep `title` and `abstract`, combine into `paper_text`, and clean whitespace/newlines.

### 2. Embedding Generation
Each paper's text is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`. Embeddings are cached to `paper_embedding.npy` to avoid recomputation.

### 3. Indexing
Embeddings are L2-normalized and added to a FAISS `IndexFlatIP` index (inner product on normalized vectors = cosine similarity). The index is cached to `paper_faiss.index`.

### 4. Search
A query is embedded the same way as the papers, normalized, and used to search the FAISS index for the top-k nearest neighbors.

```python
D, I = search_paper("deep learning for medical image analysis", k=5)
```

### 5. Summarization
Retrieved abstracts are passed through `facebook/bart-large-cnn` to generate short summaries.

### 6. Keyword Extraction
`KeyBERT` (reusing the same sentence transformer) extracts representative keywords from a paper's abstract.

## Usage

```python
# Basic semantic search
search_paper("deep learning for medical image analysis", k=5)

# Search + summarize results
search_and_summarize("deep learning for medical image analysis", k=5)

# Extract keywords from a paper
keywords = kw_model.extract_keywords(text)
```

## Setup

```bash
pip install datasets pandas sentence-transformers scikit-learn faiss-cpu
pip install transformers==4.46.3
pip install keybert==0.8.5
```

## Project Structure

```
├── 2_NLPSearchSystem.ipynb   # Main notebook (data prep, embedding, indexing, search, summarization, keywords)
├── paper_embedding.npy       # Cached paper embeddings (generated on first run)
├── paper_faiss.index         # Cached FAISS index (generated on first run)
└── README.md
```

## Future Improvements

- Hybrid search combining semantic + BM25 keyword search
- Cross-encoder re-ranking of top results for higher precision
- Evaluation with labeled queries (Precision@k, Recall@k, MRR)
- Scalable index (HNSW/IVF) for larger corpora
- Interactive demo UI (Streamlit/Gradio)
- Caching of summaries and keywords per paper

## Acknowledgments

- Dataset: [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)
- Models: [sentence-transformers](https://www.sbert.net/), [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn), [KeyBERT](https://github.com/MaartenGr/KeyBERT)
