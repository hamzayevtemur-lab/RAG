"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF).
Executes parallel Dense and Sparse (BM25) searches, then fuses
the candidate rankings using the Cormack et al. RRF formula.
"""

from dataclasses import dataclass
from chunking import AdvancedChunk
from bm25 import BM25SearchEngine, SparseSearchResult
from embeddings import DenseVectorStore, DenseSearchResult

@dataclass
class FusedCandidate:
    """A retrieved chunk enriched with rank information from both systems."""
    chunk: AdvancedChunk
    rrf_score: float
    dense_rank: int | None      # None if not present in top-K dense
    sparse_rank: int | None     # None if not present in top-K BM25
    dense_score: float | None
    sparse_score: float | None



class HybridRetriever:
    """Orchestrates Dense + BM25 search and merges via RRF."""

    def __init__(self, dense_store: DenseVectorStore, bm25_engine: BM25SearchEngine, rrf_k: int=60):
        self.dense_store = dense_store
        self.bm25_engine = bm25_engine
        self.rrf_k = rrf_k

    
    def retrieve(
        self,
        query: str,
        top_k_dense: int = 15,
        top_k_sparse: int = 15,
        top_k_fused: int = 15
    ) -> list[FusedCandidate]:

        """
        1. Run Dense semantic vector search.
        2. Run BM25 sparse keyword search.
        3. Merge candidates with Reciprocal Rank Fusion (RRF).
        """

        # Step 1: Parallel retrieval
        dense_results: list[DenseSearchResult]=self.dense_store.search(query, top_k=top_k_dense)
        sparse_results: list[SparseSearchResult]=self.bm25_engine.search(query, top_k=top_k_sparse)

        # Step 2: Accumulate RRF scores
        # Map: chunk_id -> dict with metadata & accumulated score
        candidate_map: dict[str, dict] = {}

        # Process Dense ranks
        for r in dense_results:
            cid=r.chunk.chunk_id
            rrf_contrib=1.0/(self.rrf_k+r.rank)
            candidate_map[cid]={
                "chunk": r.chunk,
                "rrf_score": rrf_contrib,
                "dense_rank": r.rank,
                "dense_score": r.score,
                "sparse_rank": None,
                "sparse_score": None,
            }

        
        # Process Sparse (BM25) ranks
        for r in sparse_results:
            cid=r.chunk.chunk_id
            rrf_contrib=1.0/(self.rrf_k+r.rank)
            if cid in candidate_map:
                candidate_map[cid]["rrf_score"] += rrf_contrib
                candidate_map[cid]["sparse_rank"] = r.rank
                candidate_map[cid]["sparse_score"] = r.score

            else:
                candidate_map[cid]={
                    "chunk": r.chunk,
                    "rrf_score": rrf_contrib,
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": r.rank,
                    "sparse_score": r.score,
                }

        
        # Step 3: Sort by descending RRF score
        sorted_candidates=sorted(
            candidate_map.values(), 
            key=lambda x:x["rrf_score"], 
            reverse=True
        )[:top_k_fused]

        fused_list=[
            FusedCandidate(
                chunk=item["chunk"],
                rrf_score=item["rrf_score"],
                dense_rank=item["dense_rank"],
                sparse_rank=item["sparse_rank"],
                dense_score=item["dense_score"],
                sparse_score=item["sparse_score"],
            )
            for item in sorted_candidates
        ]

        return fused_list
