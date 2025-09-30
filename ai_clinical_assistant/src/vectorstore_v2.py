from __future__ import annotations
import os
import json
import pprint
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import polars as pl
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Standardize file paths to a single directory ---
STORE_DIR = Path("../data/rag_store")
STORE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = STORE_DIR / "medical_knowledge.index"
EMBED_NPY = STORE_DIR / "embeddings.npy"
META_JSON = STORE_DIR / "embedding_ready.json"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "2"))


# ---------- embedding + FAISS helpers ----------
def load_embedding_model(name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    logger.info("Embedding %d texts", len(texts))
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb_norm = (emb / norms).astype("float32")
    return emb_norm


def build_or_load_store(json_data: List[Dict[str, Any]], model: SentenceTransformer, rebuild: bool = False
                        ) -> Tuple[faiss.Index, List[Dict[str, Any]], np.ndarray]:
    """
    Builds or loads a FAISS index from disk.
    """
    if INDEX_PATH.exists() and EMBED_NPY.exists() and META_JSON.exists() and not rebuild:
        logger.info("Loading vector store from disk")
        index = faiss.read_index(str(INDEX_PATH))
        embeddings = np.load(EMBED_NPY)
        metadata = json.loads(META_JSON.read_text(encoding="utf-8"))
        return index, metadata, embeddings

    logger.info("Building new FAISS index from dataset")
    texts = []
    metadata = []
    for rec in json_data:
        content = (rec.get("content") or " ".join(filter(None, [rec.get("disease"), rec.get("symptoms"), rec.get("description")])))
        content = " ".join(content.split()).strip()
        texts.append(content)
        metadata.append({
            "disease": rec.get("disease", ""),
            "symptoms": rec.get("symptoms", ""),
            "precautions": rec.get("precautions", ""),
            "description": rec.get("description", ""),
            "content": content
        })

    embeddings = embed_texts(texts, model)
    dim = embeddings.shape[1]
    logger.info("FAISS dim=%d n=%d", dim, embeddings.shape[0])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    np.save(EMBED_NPY, embeddings)
    META_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved vector store artifacts in %s", STORE_DIR)
    return index, metadata, embeddings


def faiss_search(index: faiss.Index, query_vec: np.ndarray, k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
    q = query_vec.reshape(1, -1).astype("float32")
    q = q / (np.linalg.norm(q) + 1e-12)
    D, I = index.search(q, k)
    return D[0], I[0]



# -----------------------------
# 4. QUERY FAISS VECTOR STORE
# -----------------------------
def query_vector_store(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = TOP_K
) -> Dict[str, Any]:
    """
    Queries FAISS vector store for top-K most relevant matches.
    """
    logging.info(f"🔎 Querying FAISS for: {query}")

    # Create query embedding (normalize for cosine similarity)
    qvec = model.encode([query], convert_to_numpy=True)
    # qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    # qvec = qvec.astype("float32")

    # Perform search
    distances, indices = faiss_search(index, qvec, top_k)
    results = []

    for idx, score in zip(indices, distances):
        if idx == -1:
            continue
        item = metadata[idx]
        results.append({
            **item,
            "score": float(score),
        })

    # Build context string for LLM usage
    context = "Context: ".join([
        f"\n\ncontent: {r.get('content', 'N/A')}\n"
    
        "------------------------------------------------------"
        for r in results
    ])

    return {
        "query": query,
        "results": results,
        "context": context,
    }

def load_json_data(json_path: Path) -> List[Dict[str, Any]]:
    """
    Loads JSON data from a file.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Error: The file '{json_path}' was not found.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def vector_store_setup_from_json(json_data_path: Path,  rebuild: bool = False) -> Dict[str, Any]:
    """
    High-level helper: JSON → Embeddings → FAISS → Query demo.
    """
    json_input = load_json_data(json_data_path)
    embed_model = load_embedding_model(EMBED_MODEL_NAME)
    index, metadata, embeddings = build_or_load_store(json_input, embed_model, rebuild=rebuild)

    # Example test query
    example_query = "What are the symptoms and precautions for malaria?"
    retrieval = query_vector_store(example_query, index, metadata, model=embed_model)

    logging.info("\n=== Example Retrieval ===")
    print(json.dumps(retrieval, indent=2))

    return retrieval



if __name__ == "__main__":

    vector_store_setup_from_json(META_JSON)