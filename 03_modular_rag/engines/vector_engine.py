"""
Unstructured Text Vector Engine.

Handles descriptive, qualitative narratives (e.g., risk factors,
earnings call commentary, corporate governance, ESG disclosures)
using local SentenceTransformer dense vector search.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_file: str
    score: float


@dataclass
class VectorQueryResult:
    query: str
    top_chunks: list[TextChunk]
    formatted_context: str


class VectorEngine: 
    """Indexes unstructured text documents and performs semantic search."""

    def __init__(self, model_name: str="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[VectorEngine] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        print(f"[VectorEngine] Ready — dimension: {self.dimension}")


    def index_corpus(self, data_dir: Path, chunk_size: int=600, chunk_overlap: int=80):
        """Load, chunk, and embed all .txt files from data/."""
        txt_files = sorted(data_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt documents found in {data_dir}")

        all_chunks = []
        for file_path in txt_files:
            content=file_path.read_text(encoding="utf-8")

            # Paragraph-aware split
            raw_paragraphs=content.split("\n\n")
            current=""
            for p in raw_paragraphs:
                p=p.strip()
                if not p: 
                    continue
                if len(current)+len(p)<=chunk_size:
                    current=f"{current}\n\n{p}".strip()
                else:
                    if current:
                        all_chunks.append({"text": current, "source": file_path.name})
                    current=p

            if current:
                all_chunks.append({"text": current, "source": file_path.name})

        self.chunks=all_chunks
        texts=[c["text"] for c in all_chunks]

        print(f"[VectorEngine] Embedding {len(all_chunks)} narrative chunks...")
        vectors=self.model.encode(
            texts, show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64
        )
        self.embeddings = np.array(vectors, dtype=np.float32)
        print(f"[VectorEngine] Indexed {len(self.chunks)} text chunks.")

    
    def search(self, query: str, top_k: int=3)-> VectorQueryResult:
        """Retrieve top_k most similar chunks for a narrative query."""
        if self.embeddings is None:
            raise ValueError("VectorEngine is empty. Call index_corpus() first.")

        query_vec=self.model.encode(query, normalize_embeddings=True)
        query_vec=np.array(query_vec, dtype=np.float32)

        # Cosine similarity via dot product
        scores = query_vec @ self.embeddings.T

        top_k=min(top_k, len(scores))
        top_indices=np.argpartition(scores, -top_k)[-top_k:]
        top_indices=top_indices[np.argsort(scores[top_indices])[::-1]]

        results=[]
        formatted_blocks=[]
        for rank, idx in enumerate(top_indices, 1):
            score=float(scores[idx])
            chunk_data=self.chunks[idx]

            tc = TextChunk(
                chunk_id=f"chunk_{idx}",
                text=chunk_data["text"],
                source_file=chunk_data["source"],
                score=score
            )

            results.append(tc)
            formatted_blocks.append(
                f'<chunk id="{rank}" source="{tc.source_file}" score="{score:.4f}">\n'
                f"{tc.text}\n"
                f'</chunk>'
            )

        return VectorQueryResult(
            query=query,
            top_chunks=results,
            formatted_context="\n\n".join(formatted_blocks)
        )