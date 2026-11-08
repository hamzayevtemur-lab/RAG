"""
FastAPI Server for Advanced RAG.
Exposes REST endpoints with full retrieval audit trails
and serves the interactive dashboard.
"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import config
from pipeline import AdvancedRAGPipeline, PipelineQueryResult

app=FastAPI(title="Advanced RAG Explorer")

# Initialize and index pipeline once on server startup
pipeline=AdvancedRAGPipeline(config)
pipeline.index()


# Serve static directory (HTML, CSS, JS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class QueryRequest(BaseModel):
    question: str
    use_expansion: bool = True


class RerankedChunkInfo(BaseModel):
    final_rank: int
    source: str
    cross_encoder_score: float
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    text: str


class AdvancedQueryResponse(BaseModel):
    question: str
    answer: str
    expanded_queries: list[str]
    reranked_chunks: list[RerankedChunkInfo]




@app.get("/")
def serve_index():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_file)




@app.get("/api/info")
def get_info():
    """Return pipeline configuration and model specifications."""
    return {
        "rag_type": "02_advanced_rag",
        "total_chunks": len(pipeline.chunks),
        "chunking": "Recursive Boundary-Aware",
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "sparse_model": "BM25 Okapi",
        "reranker_model": config.reranker_model,
        "rrf_k": config.rrf_k,
        "top_k_candidates": config.top_k_candidates,
        "top_k_final": config.top_k_final,
        "llm_model": config.llm_model,
    }




@app.post("/api/query", response_model=AdvancedQueryResponse)
def query_endpoint(req: QueryRequest):
    question=req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")


    try:
        # Run through complete Advanced RAG pipeline
        result: PipelineQueryResult = pipeline.query(
            question=question,
            use_expansion=req.use_expansion,
            show_details=False
        )

        formatted_chunks = [
            RerankedChunkInfo(
                final_rank=r.final_rank,
                source=r.chunk.source_file,
                cross_encoder_score=round(r.rerank_score, 4),
                rrf_score=round(r.rrf_score, 4),
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
                text=r.chunk.text
            )
            for r in result.reranked_results
        ]

        return AdvancedQueryResponse(
            question=result.question,
            answer=result.answer,
            expanded_queries=result.expanded_queries,
            reranked_chunks=formatted_chunks,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced RAG Execution Error: {str(e)}")

    

if __name__ == "__main__":
    import uvicorn
    # Use port 8001 so both Naive RAG (8000) and Advanced RAG (8001) can run side-by-side!
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)

