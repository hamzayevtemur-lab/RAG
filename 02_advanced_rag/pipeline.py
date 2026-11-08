"""
Advanced RAG Pipeline — End-to-End Orchestrator.
Combines:
1. Recursive Boundary-Aware Chunking
2. Dual-Stage Hybrid Search (Dense Vector + BM25 Okapi)
3. Reciprocal Rank Fusion (RRF)
4. Multi-Query Expansion
5. Neural Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2)
6. Grounded Generation with Provenance Metadata
"""

from dataclasses import dataclass
from config import AdvancedRAGConfig
from chunking import load_and_chunk_corpus, AdvancedChunk
from bm25 import BM25SearchEngine
from embeddings import DenseVectorStore
from hybrid_retriever import HybridRetriever, FusedCandidate
from reranker import NeuralReranker, RerankResult
from query_expansion import MultiQueryExpander
from llm import AdvancedLLMClient


@dataclass
class PipelineQueryResult:
    question: str
    expanded_queries: list[str]
    hybrid_candidates: list[FusedCandidate]
    reranked_results: list[RerankResult]
    answer: str


class AdvancedRAGPipeline:
    """Master pipeline for Advanced RAG."""

    def __init__(self, config: AdvancedRAGConfig):
        self.config=config

        print("\n" + "=" * 60)
        print("INITIALIZING ADVANCED RAG PIPELINE")
        print("=" * 60)

        # 1. Initialize local models & search engines
        self.dense_store=DenseVectorStore(config.embedding_model)
        self.bm25_engine=BM25SearchEngine()
        self.hybrid_retriever=HybridRetriever(
            self.dense_store,
            self.bm25_engine,
            rrf_k=config.rrf_k
        )
        self.reranker=NeuralReranker(
            config.reranker_model
        )

        # 2. Initialize LLM clients
        self.expander=MultiQueryExpander(config.gemini_api_key, config.llm_model)
        self.llm=AdvancedLLMClient(config.gemini_api_key, config.llm_model)

        self.chunks: list[AdvancedChunk]=[]

    def index(self):
        """
        Build both the BM25 inverted index and the Dense embedding matrix.
        """
        print("\n" + "=" * 60)
        print("INDEXING CORPUS (ADVANCED RECURSIVE CHUNKING)")
        print("=" * 60)

        # Step 1: Recursive chunking with boundary safety
        self.chunks=load_and_chunk_corpus(
            data_dir=self.config.data_dir,
            max_chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Step 2: Index BM25
        self.bm25_engine.index(self.chunks)

        # Step 3: Index Dense Embeddings
        self.dense_store.index(self.chunks)

        print(f"[Pipeline] Indexing complete: {len(self.chunks)} boundary-safe chunks ready.")
        print("=" * 60)


    def query(
        self,
        question: str,
        use_expansion: bool = True,
        show_details: bool = True
    ) -> PipelineQueryResult:
        """
        Execute full Advanced RAG pipeline.
        """
        print(f"\n{'─' * 60}")
        print(f"[Question] {question}")
        print(f"{'─' * 60}")

        # Step 1: Pre-retrieval Multi-Query Expansion
        if use_expansion:
            expanded_queries=self.expander.expand(question, num_queries=3)
            if show_details:
                print(f"[Query Expansion] Generated {len(expanded_queries)} search queries:")

                for q in expanded_queries:
                    print(f". {q}")
        
        else: expanded_queries=[question]

        # Step 2: Hybrid Retrieval across all query variations
        candidate_pool: dict[str, FusedCandidate]={}

        for q in expanded_queries:
            fused=self.hybrid_retriever.retrieve(
                query=q,
                top_k_dense=self.config.top_k_dense,
                top_k_sparse=self.config.top_k_sparse,
                top_k_fused=self.config.top_k_candidates
            )

            for cand in fused:
                cid=cand.chunk.chunk_id
                # Accumulate or take highest RRF score across query variations
                if cid not in candidate_pool or cand.rrf_score> candidate_pool[cid].rrf_score:
                    candidate_pool[cid]=cand


        # Sort candidate pool to select top_k_candidates for re-ranking
        top_candidates=sorted(
            candidate_pool.values(),
            key=lambda x: x.rrf_score,
            reverse=True
        )[:self.config.top_k_candidates]

        if show_details:
            print(f"\n[Hybrid Retrieval + RRF] Selected {len(top_candidates)} candidates for re-ranking:")
            for i, c in enumerate(top_candidates[:5], 1):
                d_str = f"#{c.dense_rank}" if c.dense_rank else "—"
                s_str = f"#{c.sparse_rank}" if c.sparse_rank else "—"
                print(f"  {i}. RRF: {c.rrf_score:.4f} | Dense: {d_str} | BM25: {s_str} | {c.chunk.source_file}")


        # Step 3: Neural Cross-Encoder Re-Ranking
        reranked = self.reranker.rerank(
            query=question,
            candidates=top_candidates,
            top_k=self.config.top_k_final
        )

        if show_details:
            print(f"\n[Cross-Encoder Re-Ranking] Top {len(reranked)} High-Precision Chunks:")
            for r in reranked:
                print(f"  Rank {r.final_rank} | Cross-Encoder: {r.rerank_score:.4f} | RRF: {r.rrf_score:.4f} | {r.chunk.source_file}")
                snippet = r.chunk.text[:120].replace("\n", " ")
                print(f"    Preview: \"{snippet}...\"")



        # Step 4: Grounded Generation
        answer = self.llm.generate(question, reranked)

        print(f"\n[Answer]\n{answer}\n")

        return PipelineQueryResult(
            question=question,
            expanded_queries=expanded_queries,
            hybrid_candidates=top_candidates,
            reranked_results=reranked,
            answer=answer
        )