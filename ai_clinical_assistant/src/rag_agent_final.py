#!/usr/bin/env python3
"""
RAG agent — final fixed:
- SentenceTransformers -> FAISS retrieval
- LangChain PromptTemplate + PydanticOutputParser
- Groq LLM wrapped into a LangChain LLM
- deterministic settings (temperature=0.0)
- robust parsing + repair heuristics + audit logging
- Pydantic v2 model_validate / model_dump usage
"""

from __future__ import annotations
import ast
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import faiss
import numpy as np
import torch
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from vectorstore_v2 import load_embedding_model, build_or_load_store, faiss_search


# LangChain imports (try safe fallbacks)
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnablePassthrough
try:
    from langchain.callbacks.tracers import LangSmithTracer
except Exception:
    LangSmithTracer = None

try:
    from langchain_core.prompts import PromptTemplate
except Exception:
    from langchain.prompts import PromptTemplate  # type: ignore

try:
    from langchain.chains import LLMChain
except Exception:
    from langchain import LLMChain  # type: ignore

try:
    from langchain.output_parsers import PydanticOutputParser
except Exception:
    PydanticOutputParser = None  # optional

try:
    from langchain.llms.base import LLM
except Exception:
    # minimal stub
    class LLM:  # type: ignore
        pass

# Optional Groq client
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "600"))
TOP_K = int(os.getenv("TOP_K", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))  # deterministic

torch.set_num_threads(1)

# ---------- Pydantic output schema ----------
class DiagnosisOutput(BaseModel):
    symptoms: str
    reasoning: str
    diagnosis: str
    treatment_suggestions: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    references: List[Dict[str, Any]]



# ---------- safe parse + repair helpers ----------
def repair_common_issues(s: str) -> str:
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"(?P<pre>[:\s\[,])'(?P<body>[^']*?)'(?P<post>[,\]\s}])",
               lambda m: f'{m.group("pre")}"{m.group("body")}"{m.group("post")}',
               s)
    s = re.sub(r",\s*([\]}])", r"\1", s)
    s = re.sub(r'(?P<k>[A-Za-z0-9_]+)\s*:', lambda m: f'"{m.group("k")}":', s)
    return s


def safe_parse_json(text: str) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None
    
    # Try strict JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting a JSON-like snippet
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        snippet = m.group(0)
        try:
            return json.loads(repair_common_issues(snippet))
        except Exception:
            pass
    
    # Final attempt with `ast.literal_eval`
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
        
    return None

# ---------- Initialize Groq LLM ----------
if not ChatGroq:
    logger.error("The langchain_groq library is not installed. Please run `pip install langchain_groq`")
    # Handle the error, perhaps exit or use a fallback
    exit()

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_NEW_TOKENS
)

# ---------- Strict few-shot prompt (encourage JSON) ----------
PROMPT_TMPL = """You are a clinical assistant for healthcare professionals.
Given candidate references and patient symptoms, RETURN EXACTLY ONE JSON OBJECT (valid JSON, double quotes) matching this schema:

{{"symptoms": "string",
  "reasoning": "1-2 sentence rationale",
  "diagnosis": "string",
  "treatment_suggestions": ["string","string"],
  "confidence": 0.0,
  "references": [{{"disease":"string","score":0.0,"evidence":"string"}}]
}}

Example output (MUST FOLLOW THIS EXACTLY):
{{"symptoms":"fever and chills","reasoning":"History and symptoms match malaria","diagnosis":"Malaria","treatment_suggestions":["Confirm with microscopy or RDT","Start ACT if confirmed"],"confidence":0.92,"references":[{{"disease":"Malaria","score":0.95,"evidence":"fever chills headache"}}]}}

Now use the Candidates below to answer. Output ONLY the JSON object (no extra text).

Candidates:
{candidates}

patient_symptoms:
{symptoms}
"""
prompt_template = PromptTemplate(input_variables=["candidates", "symptoms"], template=PROMPT_TMPL)


# ---------- Audit helper ----------
def audit_raw_output(raw: str, tag: str = "llm_raw"):
    Path("logs").mkdir(exist_ok=True)
    fn = Path("logs") / f"{tag}.log"
    with open(fn, "a", encoding="utf-8") as f:
        f.write(f"TIMESTAMP: {time.time()}\n")
        f.write(raw + "\n\n----\n\n")



def extract_content_dict(msg: AIMessage) -> Optional[Dict[str, Any]]:
    # get raw content (could be str or dict)
    raw = getattr(msg, "content", None)
    if raw is None:
        return None

    # if content is a dict and contains nested 'content'
    if isinstance(raw, dict):
        # sometimes the object is { "content": "...", "additional_kwargs": {...} }
        if "content" in raw and isinstance(raw["content"], str):
            raw_str = raw["content"]
        else:
            # if it's already a parsed dict matching our JSON output
            return raw
    elif isinstance(raw, str):
        raw_str = raw
    else:
        return None

    # try strict JSON first
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        # fallback: use your safe parser from the script (repair_common_issues / safe_parse_json)
        try:
            # replace with actual import
            parsed = safe_parse_json(raw_str)
            return parsed
        except Exception:
            return None


# ---------- Parsing helper ----------
def parse_llm_output(raw: str) -> dict:
    """Try parsing LLM output into dict. Falls back to safe_parse_json."""
    if not raw:
        return {}

    # First try strict Pydantic parser
    try:
        parser = PydanticOutputParser(pydantic_object=DiagnosisOutput)
        parsed = parser.parse(raw)
        if not isinstance(parsed, dict):
            parsed = parsed.model_dump()
        return parsed
    except Exception as e:
        logger.warning("Pydantic parse failed: %s", e)

    # Fallback: attempt to repair JSON
    parsed = safe_parse_json(raw)
    if parsed:
        return parsed

    logger.error("All parsing strategies failed.")
    return {"error": "parse_failed", "raw": raw}


# ---------- Agent flow ----------
def agent_diagnose(symptoms: str, index: faiss.Index, metadata: List[Dict[str, Any]], embeddings_model: SentenceTransformer,
                   llm: Optional[LLM], tracer: Optional[object] = None, top_k: int = TOP_K) -> DiagnosisOutput:
    # embed
    qvec = embeddings_model.encode([symptoms], convert_to_numpy=True)[0]
    # qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

    # retrieve
    D, I = faiss_search(index, qvec, top_k)
    candidates: List[Dict[str, Any]] = []
    for score, idx in zip(D, I):
        if idx < 0:
            continue
        if idx < len(metadata):
            m = metadata[idx]
            candidates.append({
                "disease": m.get("disease"),
                "score": float(score),
                "content": m.get("content", m.get("symptoms", "")),
                "symptoms": m.get("symptoms", "")
            })

    if not candidates:
        raise RuntimeError("No candidates returned by FAISS.")

    cand_text = "\n".join([f"- {c['disease']} (score={c['score']:.4f}): {c['content']}" for c in candidates])


    raw_output: str = ""

    callbacks = [tracer] if tracer is not None else None
    if llm:
        chain = (
            prompt_template
            | llm
    )
        if callbacks:
            chain = chain.with_options(callbacks=callbacks)
        try:
            res = chain.invoke({"candidates": cand_text, "symptoms": symptoms})
            print("raw_output is ",type(res))
            if isinstance(res, AIMessage):
                parsed_dict = extract_content_dict(res)
                raw_output = json.dumps(parsed_dict) if parsed_dict else res.content
            elif isinstance(res, dict):
                raw_output = json.dumps(res)
            else:
                raw_output = str(res)

        except Exception as e:
            logger.error("LLMChain.invoke failed: %s", e)
            raw_output = ""
    
    parsed = parse_llm_output(raw_output)

    if not parsed:
        audit_raw_output(raw_output)
        logger.warning("Falling back to top candidate. Raw output: %s", raw_output[:200])
        top = candidates[0]
        parsed = {
            "symptoms": symptoms,
            "reasoning": f"Top match by vector similarity: {top['disease']}",
            "diagnosis": top['disease'],
            "treatment_suggestions": [top.get("content", "Refer to clinician")],
            "confidence": float(top.get("score", 0.0)),
            "references": [{"disease": c["disease"], "score": c["score"], "evidence": c["content"]} for c in candidates]
        }


    try:
        return DiagnosisOutput.model_validate(parsed)
    except ValidationError as ve:
        logger.error("Final validation failed: %s", ve)
        top = candidates[0]
        return DiagnosisOutput(
            symptoms=symptoms,
            reasoning=f"Final validation failed, falling back. Top match: {top['disease']}",
            diagnosis=top['disease'],
            treatment_suggestions=["Check original output in logs for parsing issues."],
            confidence=0.0,
            references=[{"disease": c["disease"], "score": c["score"], "evidence": c["content"]} for c in candidates]
        )


# ---------- Demo ----------
def demo(
    symptoms: str,
    json_path: Optional[Path] = None,
    rebuild: bool = False,
    llm: Optional[LLM] = None,
    top_k: int = TOP_K,
):
    """
    Run the full RAG-based diagnosis pipeline.
    
    Args:
        symptoms: Input symptom string.
        json_path: Path to JSON dataset. Defaults to ./rag_store/embedding_ready.json.
        rebuild: If True, force rebuild of FAISS index.
        llm: Optional LLM to use for reasoning.
        top_k: Number of nearest neighbors to retrieve.
    """

    if json_path is None:
        json_path = Path("../data/rag_store/embedding_ready.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Source data file '{json_path}' missing. Place dataset there.")

    # load dataset + embedding model
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    embed_model = load_embedding_model(EMBED_MODEL_NAME)

    # build or load FAISS store
    index, metadata, embeddings = build_or_load_store(json_data, embed_model, rebuild=rebuild)

    # initialize tracer (if LangSmith is configured)
    tracer = None
    if LangSmithTracer and os.getenv("LANGCHAIN_API_KEY"):
        try:
            tracer = LangSmithTracer()
            logger.info("LangSmith tracer initialized.")
        except Exception as e:
            logger.warning("LangSmithTracer init failed: %s", e)

    # run diagnosis
    result = agent_diagnose(symptoms, index, metadata, embed_model, llm, tracer=tracer, top_k=top_k)

    # pretty print
    out = result.model_dump()
    print("\n=== RAG Agent Output (JSON) ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    return result

if __name__ == "__main__":
    user_symptoms = "persistent cough, chest pain, and shortness of breath"
    demo(user_symptoms, rebuild=False, llm=llm)


