"""
Modular RAG Synthesizer.

Fuses specialized engine outputs (SQL tabular data, Knowledge Graph
paths, narrative vector chunks, or direct parametric knowledge)
into a unified, verified response.
"""
from dataclasses import dataclass
from google import genai
from router import RouteType
from engines.tabular_engine import TabularQueryResult
from engines.graph_engine import GraphQueryResult
from engines.vector_engine import VectorQueryResult


@dataclass
class SynthesisResult:
    answer: str
    evidence_markdown: str
    route_type: RouteType


class ModularSynthesizer:
    """Produces the final response grounded in routed engine evidence."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def synthesize(
        self,
        question: str,
        route_type: RouteType,
        engine_output: TabularQueryResult | GraphQueryResult | VectorQueryResult | None
    ) -> SynthesisResult:
        """Route-specific answer formulation."""

        # ── Route 1: DIRECT (Zero Retrieval) ─────────────────
        if route_type == RouteType.DIRECT:
            prompt = f"""You are the Modular RAG Enterprise Assistant for NVIDIA technical and financial intelligence.
Respond conversationally and concisely to the user's greeting or inquiry.
Explain that you can execute:
- Deterministic SQL queries on SEC 10-K financial tables
- Multi-hop entity traversal across semiconductor supply-chain graphs
- Deep semantic vector search over technical and regulatory documents

User: {question}
Assistant:"""
            res = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return SynthesisResult(
                answer=res.text.strip(),
                evidence_markdown="*Zero database retrieval needed (Direct conversational route).* ",
                route_type=route_type
            )

        # ── Route 2: TABULAR_SQL (Deterministic Table) ───────
        elif route_type == RouteType.TABULAR_SQL:
            tab: TabularQueryResult = engine_output
            evidence = f"**Executed SQL:**\n```sql\n{tab.sql}\n```\n\n**Query Results:**\n{tab.formatted_markdown}"
            prompt = f"""You are a financial analyst. The user asked an analytical question.
A deterministic SQL query was executed against the official NVIDIA SEC 10-K database with the following results:

{evidence}

RULES:
- Answer the user's question directly using the exact numbers from the SQL result.
- Present the table cleanly in markdown.
- Do not alter or approximate any financial figures.

Question: {question}
Answer:"""
            res = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return SynthesisResult(
                answer=res.text.strip(),
                evidence_markdown=evidence,
                route_type=route_type
            )

        # ── Route 3: GRAPH_RELATIONS (Multi-Hop Paths) ───────
        elif route_type == RouteType.GRAPH_RELATIONS:
            graph: GraphQueryResult = engine_output
            evidence = f"**Traversed Knowledge Graph Subgraph:**\n{graph.formatted_paths}"
            prompt = f"""You are an enterprise hardware and supply-chain architect.
The user asked about relationships between hardware, foundries, suppliers, or regulations.
The following verified Knowledge Graph paths were traversed:

{evidence}

RULES:
- Synthesize a comprehensive answer explaining the full chain of relationships.
- Explicitly mention the key entities (e.g. suppliers, foundries, packaging technologies, chips) and how they connect.
- Base your claims strictly on the provided graph paths.

Question: {question}
Answer:"""
            res = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return SynthesisResult(
                answer=res.text.strip(),
                evidence_markdown=evidence,
                route_type=route_type
            )

        # ── Route 4: VECTOR_SEARCH (Narrative Text) ──────────
        else:
            vec: VectorQueryResult = engine_output
            evidence = vec.formatted_context
            prompt = f"""You are an enterprise research analyst.
Answer the user's question using ONLY the retrieved narrative context chunks below.

<context>
{evidence}
</context>

RULES:
- Answer strictly from the provided text chunks.
- Cite your sources using bracketed chunk IDs, e.g., [Chunk 1].
- If information is insufficient, state that clearly.

Question: {question}
Answer:"""
            res = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return SynthesisResult(
                answer=res.text.strip(),
                evidence_markdown=evidence,
                route_type=route_type
            )