"""
Naive RAG Pipeline — End-to-End Orchestrator.

Flow:
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Load    │───▶│  Chunk   │───▶│  Embed   │───▶│  Search  │───▶│ Generate │
    │  Files   │    │  (fixed) │    │ (local)  │    │ (NumPy)  │    │ (Gemini) │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
       5 files       ~230 chunks    (N, 384) matrix   top_k results   final answer

Two phases:
    1. INDEXING (run once): Load → Chunk → Embed → Store
    2. QUERYING (run many): Question → Embed → Search → Generate
"""

from config import NaiveRAGConfig
from chunking import load_and_chunk_all
from embeddings import EmbeddingEngine
from vector_store import VectorStore, SearchResult
from llm import LLMClient



class NaiveRAGPipeline:
    """Complete Naive RAG system wired together."""

    def __init__(self, config: NaiveRAGConfig):
        self.config=config

        # Initialize components
        self.embedder = EmbeddingEngine(config.embedding_model)
        self.store = VectorStore()
        self.llm = LLMClient(config.gemini_api_key, config.llm_model)

    def index(self):
        """
        INDEXING PHASE — run once to build the searchable knowledge base.

        Steps:
            1. Read all .txt files from data/
            2. Split into fixed-size chunks
            3. Embed every chunk into a 384-dim vector
            4. Store in the vector store
        """

        print("\n" + "=" * 60)
        print("INDEXING PHASE")
        print("=" * 60)

        # Step 1 & 2: Load and chunk all documents
        chunks=load_and_chunk_all(
            data_dir=self.config.data_dir,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        # Step 3: Embed all chunks
        print(f"\n[Pipeline] Embedding {len(chunks)} chunks... ")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Step 4: Store in vector store
        self.store.add(chunks, embeddings)

        print(f"\n[Pipeline] Indexing complete — {len(self.store)} chunks ready")
        print("=" * 60)

    
    def query(self, question: str, show_context: bool=True)-> str:
        """
        QUERYING PHASE — answer a question using the indexed knowledge base.

        Steps:
            1. Embed the question into a vector
            2. Search for top_k most similar chunks
            3. Send chunks + question to Gemini for generation

        Args:
            question:     The user's question.
            show_context: If True, print retrieved chunks for inspection.

        Returns:
            The generated answer.
        """

        print(f"\n{'─' * 60}")
        print(f"Question: {question}")
        print(f"{'─' * 60}")

        # Step 1: Embed the question
        query_vector=self.embedder.embed_query(question)

        # Step 2: Retrieve top_k chunks
        results: list[SearchResult]=self.store.search(query_vector, self.config.top_k)

        # Display retrieved chunks (for debugging and understanding)
        if show_context:
            print(f"\n[Retrieval] Top {self.config.top_k} chunks:")
            for i, result in enumerate(results, 1):
                print(f"\n  Chunk {i} | score: {result.score:.4f} | "
                      f"source: {result.chunk.source_file}")
                    
                # Show first 150 chars as preview
                preview = result.chunk.text[:150].replace("\n", " ")
                print(f"  Preview: {preview}...")

        # Step 3: Build context and generate answer
        context_chunks=[
            {
                "text": r.chunk.text,
                "source": r.chunk.source_file,
                "score": r.score,
            }
            for r in results
        ]

        answer = self.llm.generate(question, context_chunks)

        print(f"\n[Answer]\n{answer}")
        
        return answer