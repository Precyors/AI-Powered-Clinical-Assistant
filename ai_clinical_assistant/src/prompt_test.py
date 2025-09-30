import json
import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any, Union
from vector_store import query_vector_store
from dotenv import load_dotenv
import os   

load_dotenv()


# ====================================
# 1. BUILD FEW-SHOT EXAMPLES
# ====================================
FEW_SHOT_EXAMPLES = """
Example 1:
Disease: Malaria
Symptoms: fever, chills, headache, sweating
Precautions: use mosquito nets, take antimalarial drugs, avoid stagnant water
Description: Malaria is a life-threatening disease transmitted through the bite of infected mosquitoes.

Example 2:
Disease: Asthma
Symptoms: wheezing, chest tightness, shortness of breath, coughing
Precautions: avoid triggers, use inhalers, maintain air quality, seek regular check-ups
Description: Asthma is a chronic lung condition where the airways become inflamed, narrow, and produce extra mucus.

Example 3:
Disease: Hypertension
Symptoms: headache, dizziness, blurred vision, chest pain
Precautions: reduce salt intake, exercise regularly, avoid alcohol, manage stress
Description: Hypertension is a condition where blood pressure remains elevated, increasing the risk of heart disease and stroke.
"""

# ====================================
# 2. RAG PIPELINE FUNCTION
# ====================================
def rag_with_opengpt_oss(
    query: Union[str, Dict[str, str]],
    index,
    embedding_ready: List[Dict[str, Any]],
    model: SentenceTransformer,
    openrouter_api_key: str,
    top_k: int = 3,
    temperature: float = 0.2
) -> str:
    """
    Full RAG pipeline:
      - Retrieves relevant context from FAISS
      - Uses few-shot examples for better medical reasoning
      - Accepts structured inputs (symptoms, labs, etc.)
      - Sends query + context to OpenGPT-OSS via OpenRouter
    """

    # ✅ If user provides structured input, build a natural language query
    if isinstance(query, dict):
        structured_parts = []
        for key, value in query.items():
            structured_parts.append(f"{key.capitalize()}: {value}")
        query_text = "\n".join(structured_parts)
    else:
        query_text = query

    # ✅ Retrieve top-k matches from FAISS
    retrieval = query_vector_store(query_text, index, embedding_ready, model, top_k)
    context = retrieval["context"]

    # ✅ Build a powerful prompt with few-shot examples + structured context
    prompt = f"""
You are a highly knowledgeable AI medical assistant.
Use the retrieved medical context and examples to provide the most accurate response.

Here are a few medical reasoning examples for reference:
{FEW_SHOT_EXAMPLES}

Retrieved Context:
{context}

User's Case:
{query_text}

Answer step-by-step, using evidence from the context first.
If the answer is not in the context, respond with:
"I don't have enough information from my knowledge base."
    """

    # ✅ Send request to OpenRouter API
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [
            {"role": "system", "content": "You are a medical assistant powered by RAG."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,  # 🔹 Control creativity level dynamically
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()

    # ✅ Extract model's answer
    try:
        answer = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        answer = f"Error from OpenRouter: {response_json}"

    return answer


# ====================================
# 3. USAGE EXAMPLES
# ====================================
if __name__ == "__main__":
    # Load processed knowledge base
    with open("./embeddings_ready/embedding_ready.json", "r") as f:
        embedding_ready = json.load(f)

    # Load FAISS index
    index = faiss.read_index("./faiss_index/medical_knowledge.index")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # OpenRouter API Key
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

     # EXAMPLE 1 → Free-text query
    print("\n=== SIMPLE QUERY ===")
    answer = rag_with_opengpt_oss(
        query="What precautions should I take for pneumonia?",
        index=index,
        embedding_ready=embedding_ready,
        model=model,
        openrouter_api_key=OPENROUTER_API_KEY,
        top_k=3,
        temperature=0.3
    )
    print(answer)

    # EXAMPLE 2 → Structured input query
    print("\n=== STRUCTURED QUERY ===")
    structured_query = {
        "symptoms": "hip_joint_pain, knee_pain, swelling_joints",
        "labs": "cartilage wear and tear"
    }
    answer = rag_with_opengpt_oss(
        query=structured_query,
        index=index,
        embedding_ready=embedding_ready,
        model=model,
        openrouter_api_key=OPENROUTER_API_KEY,
        top_k=3,
        temperature=0.1
    )
    print(answer)