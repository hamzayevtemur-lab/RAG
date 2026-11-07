"""
Local Embedding Engine using SentenceTransformers.

Loads all-MiniLM-L6-v2 (~80 MB) on your M1 Mac.
Converts text into 384-dimensional dense vectors
where semantically similar text → nearby vectors.

Architecture: 6-layer BERT encoder → mean pooling → L2 normalize
Parameters:   22.7 million (vs 175B for GPT-3 — this is tiny)
Output:       384-dimensional unit vector per input text
"""

import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    """Wraps SentenceTransformer for encoding text into dense vectors."""

    def __init__(self, model_name: str):
        """
        Load the embedding model.
        On first run, this downloads ~80 MB from HuggingFace.
        Subsequent runs load from local cache instantly.
        """
        print(f"[Embeddings] Loading model: {model_name}")
        self.model=SentenceTransformer(model_name)

        self.dimension=self.model.get_embedding_dimension()
        print(f"[Embeddings] Ready — dimension: {self.dimension}")

    def embed_texts(self, texts: list[str])->np.ndarray:
        """
        Encode a list of texts into a matrix of vectors.
        Args:
            texts: List of N text strings.
        Returns:
            NumPy array of shape (N, 384) — one vector per text.
            Each vector is L2-normalized (unit length) so that
            dot product == cosine similarity.
        """
        vectors=self.model.encode(
            texts, show_progress_bar=True, 
            normalize_embeddings=True,    # L2 normalize → dot product = cosine sim
            batch_size=64,                # Process 64 texts at a time (memory safe)
        )

        return np.array(vectors, dtype=np.float32)


    def embed_query(self, query: str)-> np.ndarray:
        """
        Encode a single query into a vector.
        Args:
            query: The user's question.
        Returns:
            NumPy array of shape (384,) — a single unit vector.
        """
        vector=self.model.encode(
            query, 
            normalize_embeddings=True
        )
        return np.array(vector, dtype=np.float32)




