"""
Modular RAG Configuration.
Hyperparameters for Semantic Routing, SQLite Tabular Engine,
Entity-Relation Graph Engine, and Hybrid Vector Engine.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class ModularRAGConfig:
    # ── Shared Data Directory ─────────────────────────────
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    db_path: Path = Path(__file__).resolve().parent.parent / "data" / "nvidia_financials.db"

    # ── Models ───────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gemini-3.1-flash-lite"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # ── Modular Routing Thresholds & Limits ──────────────
    vector_top_k: int = 3
    graph_max_hops: int = 2
    web_port: int = 8002



config=ModularRAGConfig()