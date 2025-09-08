# ai_clinical_assistant/data/vector_store.py
from __future__ import annotations
import os
import json
import pprint
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import polars as pl
import faiss
from sentence_transformers import SentenceTransformer
from clean import clean_medical_data, embedding_ready, merge_datasets 

# -----------------------------
# CONFIG
# -----------------------------


INDEX_PATH = Path("./faiss_index/medical_knowledge.index")
EMBEDDINGS_PATH = Path("./numpy_file/embeddings.npy")
EMBEDDING_READY = Path("./embeddings_ready/embedding_ready.json")

EMBEDDING_MODEL =  "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")



def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Loads the embedding model.
    """
    logging.info(f"🔄 Loading embedding model: {model_name}...")
    return SentenceTransformer(model_name)


def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Generates normalized embeddings for a list of texts.
    """
    logging.info(f"🔄 Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return (embeddings / norms).astype("float32")  # Normalize for cosine similarity

def load_json_data(json_path: Path) -> List[Dict[str, Any]]:
    """
    Loads JSON data from a file.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Error: The file '{json_path}' was not found.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# 3. CREATE FAISS VECTOR STORE
# -----------------------------
def create_vector_store(json_data: List[Dict[str, Any]], model_name: str = EMBEDDING_MODEL) -> Tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    """
    Takes raw medical data → cleans → embeds → stores in FAISS.
    Returns (faiss_index, cleaned_metadata, embedding_model).
    """
    logging.info("🔄 Preparing json data for embedding...")

    # Prepare texts for embedding
    texts = [each_data["content"] for each_data in json_data if "content" in each_data and isinstance(each_data["content"], str)]
    # Load embedding model
    model = load_embedding_model(model_name)

    # Generate embeddings
    embeddings = generate_embeddings(texts, model)
    dim = embeddings.shape[1]

    # Create FAISS index (cosine similarity)
    logging.info(f"🔄 Creating FAISS index with dimension={dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist artifacts
    faiss.write_index(index, str(INDEX_PATH))
    np.save(EMBEDDINGS_PATH, embeddings)
    # with open(METADATA_PATH, "w", encoding="utf-8") as f:
    #     json.dump(cleaned_data, f, indent=2)

    logging.info(f"✅ Vector store created with {index.ntotal} entries.")
    return index, json_data, model


# -----------------------------
# 4. QUERY FAISS VECTOR STORE
# -----------------------------
def query_vector_store(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Queries FAISS vector store for top-K most relevant matches.
    """
    logging.info(f"🔎 Querying FAISS for: {query}")

    # Create query embedding (normalize for cosine similarity)
    qvec = model.encode([query], convert_to_numpy=True)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    qvec = qvec.astype("float32")

    # Perform search
    distances, indices = index.search(qvec, top_k)
    results = []

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        results.append({
            **item,
            "score": float(score),
        })

    # Build context string for LLM usage
    context = "Context: \n\n".join([
        f"content: {r.get('content', 'N/A')}\n"
    
        "------------------------"
        for r in results
    ])

    return {
        "query": query,
        "results": results,
        "context": context,
    }


def vector_store_setup_from_json(json_data_path: Path) -> Dict[str, Any]:
    """
    High-level helper: JSON → Embeddings → FAISS → Query demo.
    """
    json_input = load_json_data(json_data_path)
    index, metadata, model = create_vector_store(json_input)

    # Example test query
    example_query = "What are the symptoms and precautions for malaria?"
    retrieval = query_vector_store(example_query, index, metadata, model)

    logging.info("\n=== Example Retrieval ===")
    print(json.dumps(retrieval, indent=2))

    return retrieval


if __name__ == "__main__":

    vector_store_setup_from_json(EMBEDDING_READY)
   








# # ===============================
# # 5. QUERY VECTOR STORE
# # ===============================
# def query_vector_store(
#     query: str,
#     index,
#     embedding_ready: List[Dict[str, Any]],
#     model: SentenceTransformer,
#     top_k: int = 3
# ) -> Dict[str, Any]:
#     """
#     Queries the FAISS vector store and retrieves top-k matches.
#     """
#     query_embedding = model.encode(query).astype("float32").reshape(1, -1)
#     distances, indices = index.search(query_embedding, top_k)

#     results = []
#     for idx, distance in zip(indices[0], distances[0]):
#         if idx == -1:  # FAISS returns -1 when there’s no match
#             continue
#         item = embedding_ready[idx]
#         item_with_score = {**item, "distance": float(distance)}
#         results.append(item_with_score)

#     results = sorted(results, key=lambda x: x["distance"])
#     context_chunks = []
#     for r in results:
#         context_chunks.append(
#             f"Disease: {r.get('disease', 'N/A')}\n"
#             f"Symptoms: {r.get('symptoms', 'N/A')}\n"
#             f"Precautions: {r.get('precautions', 'N/A')}\n"
#             f"Description: {r.get('description', 'N/A')}\n"
#             "------------------------"
#         )
#     context = "\n".join(context_chunks)

#     # 6. Return results + context
#     return {
#         "query": query,
#         "results": results,
#         "context": context
#     }



# # ===============================
# # 6. MAIN ENTRY POINT
# # ===============================
# def vector_store_setup_from_json(json_input: Dict[str, Any]):
#     """
#     High-level pipeline: JSON -> Embeddings -> FAISS -> Query.
#     """
#     index, metadata, model = create_vector_store(json_input)

#     # Example query
#     example_query = "What are the symptoms and precautions for malaria?"
#     results = query_vector_store(example_query, index, metadata, model)

#     print("\n--- Example Retrieval ---")

#     # 5. Prepare a clean context string for LLM consumption
#     context_chunks = []
#     for r in results:
#         context_chunks.append(
#             f"Disease: {r.get('disease', 'N/A')}\n"
#             f"Symptoms: {r.get('symptoms', 'N/A')}\n"
#             f"Precautions: {r.get('precautions', 'N/A')}\n"
#             f"Description: {r.get('description', 'N/A')}\n"
#             "------------------------"
#         )
#     context = "\n".join(context_chunks)

#     # Return results + context
#     return results, context
 

# # ===============================
# # USAGE EXAMPLE
# # ===============================
# if __name__ == "__main__":
#     # Load JSON input
#     with open("./embeddings_ready/embedding_ready.json", "r") as f:
#         json_data = json.load(f)
#     for x,y in vector_store_setup_from_json(json_data):
#         pprint.pprint(x, indent=2)
#         pprint.pprint(y, indent=2)
#         print("\n")
