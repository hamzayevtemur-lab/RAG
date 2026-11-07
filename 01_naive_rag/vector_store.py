"""
In-Memory Vector Store — Pure NumPy Implementation.

Stores chunk embeddings as a dense matrix and performs
retrieval via matrix multiplication (dot product = cosine
similarity when vectors are L2-normalized).

How it works:
    query_vector (1, 384)  @  chunk_matrix (384, N)  =  scores (1, N)
    
    One matrix multiply instantly computes similarity 
    against ALL chunks simultaneously. This is the same
    operation as a single linear layer in a neural network.
"""

import numpy as np
from dataclasses import dataclass
from chunking import Chunk

@dataclass
class SearchResult:
    """A single retrieval result with its similarity score."""
    chunk: Chunk
    score: float          # Cosine similarity (range: -1 to +1)


class VectorStore:
    """Stores embeddings and performs cosine similarity search."""
    def __init__(self):
        self.chunks: list[Chunk]=[]
        self.embeddings: np.ndarray | None=None # Shape: (N, 384)

    
    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        """
        Index chunks and their embeddings.
        Args:
            chunks:     List of N Chunk objects.
            embeddings: NumPy array of shape (N, 384), L2-normalized.
        """
        assert len(chunks) == embeddings.shape[0], \
            f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"

        self.chunks=chunks
        self.embeddings=embeddings
        print(f"[VectorStore] Indexed {len(chunks)} chunks — matrix shape: {embeddings.shape}")


    def search(self, query_vector: np.ndarray, top_k: int)->list[SearchResult]:
        """
        Find the top_k most similar chunks to the query.

        Math:
            scores = query_vector · embeddings^T
            
            Because both are L2-normalized:
            dot_product(a, b) = cos(θ) = cosine_similarity(a, b)
            
            Score of 1.0 = identical meaning
            Score of 0.0 = completely unrelated
            Score < 0.0  = opposite meaning (rare in practice)
        Args:
            query_vector: Shape (384,) — the embedded user question.
            top_k:        How many results to return.
        Returns:
            List of top_k SearchResult objects, sorted by descending score.
        """
        # Core operation: one matrix multiply scores all chunks at once
        # (384,) @ (N, 384).T = (N,)
        scores=query_vector @ self.embeddings.T

        # Get indices of top_k highest scores
        # argpartition is O(N) vs O(N log N) for full sort — faster for large N
        top_indices=np.argpartition(scores, -top_k)[-top_k:]


        # Sort only those top_k by score (descending)
        top_indices=top_indices[np.argsort(scores[top_indices])[::-1]]

        results=[]
        for idx in top_indices:
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
            ))

        return results

    
    def __len__(self)-> int:
        return len(self.chunks)
    
