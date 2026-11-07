"""
FastAPI Web Server for Naive RAG.
Exposes REST endpoints and serves the interactive web dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from config import config
from pipeline import NaiveRAGPipeline


app = FastAPI(title="Naive RAG Explorer")

# Initialize and index pipeline once on server startup
pipeline = NaiveRAGPipeline(config)
pipeline.index()

# Serve static files (HTML, CSS, JS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class QueryRequest(BaseModel):
    question: str


class ChunkInfo(BaseModel):
    id: int
    source: str
    score: float
    text: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks: list[ChunkInfo]



@app.get("/")
def serve_index():
    index_file=static_dir/"index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_file)

@app.get("/api/info")
def get_info():
    """Return pipeline configuration and corpus statistics."""
    return {
        "rag_type": "01_naive_rag",
        "total_chunks": len(pipeline.store),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "embedding_model": config.embedding_model,
        "llm_model": config.llm_model,
    }



@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):

    question=req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Embed question and retrieve top_k chunks
    query_vector = pipeline.embedder.embed_query(question)
    results = pipeline.store.search(query_vector, config.top_k)

    context_chunks = [
        {"text": r.chunk.text, "source": r.chunk.source_file, "score": r.score}
        for r in results
    ]

    # 2. Call LLM with grounded context
    try:
        answer = pipeline.llm.generate(question, context_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation Error: {str(e)}")

    # 3. Format response with chunk inspector data
    chunk_details = [
        ChunkInfo(
            id=i,
            source=r.chunk.source_file,
            score=round(r.score, 4),
            text=r.chunk.text,
        )
        for i, r in enumerate(results, 1)
    ]

    return QueryResponse(
        question=question,
        answer=answer,
        chunks=chunk_details,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

    