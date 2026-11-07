"""
Naive RAG Configuration.
All hyperparameters and paths defined here.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class NaiveRAGConfig:
    # ── Data ──────────────────────────────────────────────
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # ── Chunking (fixed-size, no semantic awareness) ─────
    chunk_size: int = 500          # characters per chunk
    chunk_overlap: int = 50        # overlap between adjacent chunks

    # ── Embedding (runs 100% locally on M1 Mac, ~80 MB) ─
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Retrieval ────────────────────────────────────────
    top_k: int = 3

    # ── LLM Generation (cloud API) ──────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = "gemini-3.1-flash-lite"




config = NaiveRAGConfig()
