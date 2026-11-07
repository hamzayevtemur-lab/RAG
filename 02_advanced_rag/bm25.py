"""
Sparse Lexical Search Engine using BM25 Okapi.

Provides exact token and keyword matching for acronyms,
hardware model names (e.g., H200, GB200, B200), financial numbers,
and regulatory identifiers.
"""

from dataclasses import dataclass
import re
import numpy as np
from rank_bm25 import BM25Okapi

from chunking import AdvancedChunk


@dataclass
class SparseSearchResult:
    chunk: AdvancedChunk
    score: float
    rank: int



def tokenize(text: str)-> list[str]:
    """
    Simple, fast alphanumeric tokenizer.
    Converts text to lowercase tokens, preserving numbers and symbols.
    """
    return re.findall(r"\b\w+\b", text.lower())


class BM25SearchEngine:
    """In-memory BM25 Okapi index for lexical search."""

    def __init__(self):
        self.chunks: list[AdvancedChunk] = []
        self.bm25: BM25Okapi | None=None
        self.corpus_tokens: list[list[str]]=[]

    
    def index(self, chunks: list[AdvancedChunk]):
        """
        Build the BM25 inverted index from the corpus.
        """
        self.chunks=chunks
        self.corpus_tokens=[tokenize(c.text) for c in chunks]
        self.bm25=BM25Okapi(self.corpus_tokens)
        print(f"[BM25] Indexed {len(chunks)} chunks with BM25 Okapi.")

    def search(self, query: str, top_k: int)-> list[SparseSearchResult]:
        """
        Search the BM25 index for exact keyword matches.

        Args:
            query: User's question or search phrase.
            top_k: Number of highest-scoring chunks to retrieve.
        Returns:
            List of SparseSearchResult sorted by descending BM25 score.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index has not been initialized. Call .index() first.")
        
        query_tokens=tokenize(query)
        if not query_tokens:
            return []

        # Compute raw BM25 scores across all chunks
        scores=self.bm25.get_scores(query_tokens)

        # Retrieve top_k highest score indices
        top_k=min(top_k, len(scores))
        top_indices=np.argsort(scores)[::-1][:top_k]

        results=[]
        for rank, idx in enumerate(top_indices, 1):
            results.append(SparseSearchResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank
            ))

        return results

        
