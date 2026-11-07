"""
Dense Semantic Vector Search Engine.

Uses all-MiniLM-L6-v2 (local, ~80 MB) to produce 384-dimensional
dense embeddings and performs cosine similarity search via NumPy.
"""

from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import AdvancedChunk

@dataclass
class DenseSearchResult:
    chunk: AdvancedChunk
    score: float       # Cosine similarity (-1.0 to 1.0)
    rank: int          # 1-based rank position in dense results



class DenseVectorStore:
    """In-memory dense vector store with cosine similarity retrieval."""

    def __init__(self, model_name: str):
        print(f"[DenseStore] Loading embedding model: {model_name}")
        self.model=SentenceTransformer(model_name)
        self.dimension=self.model.get_embedding_dimension()
        self.chunks: list[AdvancedChunk]=[]
        self.embeddings: np.ndarray | None=None
        print(f"[DenseStore] Ready — dimension: {self.dimension}")

    def index(self, chunks: list[AdvancedChunk]):
        """
        Embed all chunks into a normalized dense matrix (N, 384).
        """
        self.chunks=chunks
        texts=[c.text for c in chunks]

        print(f"[DenseStore] Embedding {len(chunks)} chunks...")
        vectors=self.model.encode(
            texts, show_progress_bar=True,
            normalize_embeddings=True, # L2 normalize -> dot product = cosine similarity
            batch_size=64
        )

        self.embeddings=np.array(vectors, dtype=np.float32)
        print(f"[DenseStore] Dense matrix indexed: {self.embeddings.shape}")

    
    def search(self, query: str, top_k: int)->list[DenseSearchResult]:
        """
        Retrieve top_k most semantically similar chunks.
        """
        if self.embeddings is None:
            raise ValueError("Store is empty.Call index() first")

        query_vector = self.model.encode(query, normalize_embeddings=True)
        query_vector = np.array(query_vector, dtype=np.float32)

        # Matrix multiply: (384,) @ (N, 384).T -> (N,)
        scores = query_vector @ self.embeddings.T

        top_k=min(top_k, len(scores))
        top_indices=np.argpartition(scores, -top_k)[-top_k:]
        top_indices=top_indices[np.argsort(scores[top_indices])[::-1]]

        results=[]
        for rank, idx in enumerate(top_indices, 1):
            results.append(DenseSearchResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank
            ))

        
        return results

