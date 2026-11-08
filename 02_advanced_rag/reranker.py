"""
Neural Cross-Encoder Re-Ranker.

Uses ms-marco-MiniLM-L-6-v2 (local, ~80 MB) to perform
deep pairwise cross-attention between (Query, Document).
Eliminates semantic false-positives retrieved by the first stage.
"""

from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from hybrid_retriever import FusedCandidate
from chunking import AdvancedChunk


@dataclass
class RerankedResult:
    """A final candidate with cross-encoder relevance score."""
    chunk: AdvancedChunk
    rerank_score: float         # Cross-encoder output logit / probability
    final_rank: int
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None

RerankResult = RerankedResult


class NeuralReranker:
    """Re-ranks candidate chunks using a trained cross-encoder model."""
    def __init__(self, model_name: str="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Loading CrossEncoder: {model_name}")
        self.model = CrossEncoder(model_name)
        print("[Reranker] Ready.")

    def rerank(
        self, query: str, candidates: list[FusedCandidate],
        top_k: int=3
    )-> list[RerankResult]:
        """
        Score all (query, chunk_text) pairs through the cross-encoder
        and return the top_k most relevant chunks.
        """

        if not candidates:
            return []

        # Prepare (query, document) pairs
        pairs=[(query, cand.chunk.text) for cand in candidates]

        # Predict cross-encoder relevance scores
        scores=self.model.predict(pairs, show_progress_bar=False)

        # Attach score to candidates metadata
        scored_candidates=[]
        for cand, score in zip(candidates, scores):
            scored_candidates.append({
                "candidate": cand,
                "score": float(score),
            })

        # Sort by cross-encdder score descending
        sorted_candidates=sorted(
            scored_candidates, 
            key=lambda x: x["score"],
            reverse=True
        )[: top_k]

        # Convert to typed dataclass
        results=[]
        for rank, item in enumerate(sorted_candidates, 1):
            cand: FusedCandidate=item["candidate"]
            results.append(RerankResult(
                chunk=cand.chunk,
                rerank_score=item["score"],
                final_rank=rank,
                rrf_score=cand.rrf_score,
                dense_rank=cand.dense_rank,
                sparse_rank=cand.sparse_rank,
            ))

        return results
