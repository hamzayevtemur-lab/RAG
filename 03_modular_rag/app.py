"""
FastAPI Server for Modular RAG.
Exposes REST endpoints with full intent-routing audit trails
and serves the interactive dashboard on Port 8002.
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import config
from pipeline import ModularRAGPipeline, ModularQueryResult

app = FastAPI(title="Modular RAG Explorer")

# Initialize and index pipeline once on server startup
pipeline = ModularRAGPipeline(config)
pipeline.index()

# Serve static directory (HTML, CSS, JS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class QueryRequest(BaseModel):
    question: str


class RoutingInfo(BaseModel):
    route: str
    target_engine: str
    reasoning: str


class ModularQueryResponse(BaseModel):
    question: str
    routing: RoutingInfo
    evidence_markdown: str
    answer: str


@app.get("/")
def serve_index():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(index_file)



@app.get("/api/info")
def get_info():
    """Return pipeline configuration and active retrieval engines."""
    return {
        "rag_type": "03_modular_rag",
        "engines": [
            {"name": "SQLite Tabular Engine", "type": "Structured Text-to-SQL", "target": "nvidia_financials.db"},
            {"name": "Knowledge Graph Engine", "type": "Entity-Relation Graph", "triples_count": len(pipeline.graph_engine.all_triples)},
            {"name": "Vector Engine", "type": "Hybrid Text Search", "chunks_count": len(pipeline.vector_engine.chunks)},
            {"name": "Direct Route", "type": "Parametric Generation", "cost": "0 database lookups"}
        ],
        "router_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "port": config.web_port
    }



@app.post("/api/query", response_model=ModularQueryResponse)
def query_endpoint(req: QueryRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        result: ModularQueryResult = pipeline.query(question, show_details=False)
        return ModularQueryResponse(
            question=result.question,
            routing=RoutingInfo(
                route=result.routing.route.value,
                target_engine=result.routing.target_engine,
                reasoning=result.routing.reasoning
            ),
            evidence_markdown=result.evidence_markdown,
            answer=result.answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modular RAG Execution Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=config.web_port, reload=False)

