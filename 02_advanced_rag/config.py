"""
Advanced RAG Configuration.
Hyperparameters for Recursive Chunking, Hybrid Search (Dense + BM25),
Reciprocal Rank Fusion (RRF), and Cross-Encoder Re-Ranking.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AdvancedRAGConfig:
    # ── Shared Data Directory ─────────────────────────────
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # ── Recursive / Semantic Chunking ────────────────────
    chunk_size: int = 600           # Target character size per chunk
    chunk_overlap: int = 100        # Overlap to preserve inter-sentence context

    # ── Local Embedding Model (Dense Bi-Encoder) ─────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Local Cross-Encoder (Neural Re-Ranker, ~80 MB) ──
    # Reads (Query, Document) simultaneously with full cross-attention
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Hybrid Retrieval & Fusion Hyperparameters ────────
    # Stage 1: Broad Retrieval (Candidate Gathering)
    top_k_dense: int = 15           # Top K from vector similarity
    top_k_sparse: int = 15          # Top K from BM25 Okapi

    # Reciprocal Rank Fusion constant (standard industry baseline = 60)
    rrf_k: int = 60

    # Stage 2: Deep Cross-Encoder Re-Ranking
    top_k_candidates: int = 15      # Candidates passed to the cross-encoder
    top_k_final: int = 3            # Final precision chunks injected into LLM

    # ── LLM Settings (Cloud API) ─────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = "gemini-3.1-flash-lite"


config=AdvancedRAGConfig()

