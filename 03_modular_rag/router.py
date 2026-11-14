"""
Semantic Intent Router.

Classifies incoming queries into one of 4 execution routes:
1. DIRECT          -> No retrieval needed (chit-chat, system capabilities)
2. TABULAR_SQL     -> Structured queries on financial & executive tables
3. GRAPH_RELATIONS -> Multi-hop supply chain & hardware entity paths
4. VECTOR_SEARCH   -> Narrative text search (risk factors, commentary, ESG)
"""

from enum import Enum
from dataclasses import dataclass
import json
import re
from google import genai

class RouteType(str, Enum):
    DIRECT = "DIRECT"
    TABULAR_SQL = "TABULAR_SQL"
    GRAPH_RELATIONS = "GRAPH_RELATIONS"
    VECTOR_SEARCH = "VECTOR_SEARCH"


@dataclass
class RoutingDecision:
    route: RouteType
    reasoning: str
    target_engine: str


class SemanticRouter:
    """Classifies user query intent using structured LLM routing."""

    def __init__(self, api_key: str, model_name: str="gemini-3.1-flash-lite"):
        self.client=genai.Client(api_key=api_key)
        self.model_name=model_name
        print("[SemanticRouter] Initialized.")

    def route_query(self, user_question: str)-> RoutingDecision:
        """Analyze question and return the optimal execution route."""
        # 1. Fast-path heuristic for greetings to save API calls
        lower_q=user_question.strip().lower()
        if lower_q in ("hi", "hello", "hey", "who are you", "what can you do", "help"):
            return RoutingDecision(
                route=RouteType.DIRECT,
                reasoning="General greeting or inquiry about system capabilities.",
                target_engine="Parametric Direct Generator"
            )

        # 2. LLM Intent Classifier
        prompt = f"""You are an expert query router in a Modular RAG enterprise system.
Analyze the user's question and classify it into EXACTLY ONE of the following 4 routes:
1. TABULAR_SQL:
   - For questions about financial numbers, balance sheets, multi-year revenues, percentages, sums, rankings, or executive compensation tables.
   - Available SQL tables: 'specialized_market_revenue', 'executive_compensation', 'geographic_revenue'.
2. GRAPH_RELATIONS:
   - For questions about relationships between entities: hardware architectures (Hopper, Blackwell, Grace), memory suppliers (SK Hynix, Samsung, Micron), foundries (TSMC, SMIC), advanced packaging (CoWoS), export regulations (USG, China, BIS), or automotive OEM partners (Mercedes, BYD).
3. VECTOR_SEARCH:
   - For questions about qualitative narratives: legal proceedings, patent litigation, risk factors, CEO commentary from earnings calls, climate/ESG goals, or CUDA software libraries.
4. DIRECT:
   - For general greetings, meta questions, or questions requiring no enterprise database lookup.
Respond with ONLY valid JSON in this exact structure:
{{
  "route": "TABULAR_SQL" | "GRAPH_RELATIONS" | "VECTOR_SEARCH" | "DIRECT",
  "reasoning": "one concise sentence explaining why this route was chosen"
}}
User Question: {user_question}
JSON Output:""" 
        try:
            response=self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            raw=response.text.strip()
            # Clean possible markdown wrapping
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(cleaned)

            chosen_route = RouteType(data.get("route", "VECTOR_SEARCH"))
            reasoning = data.get("reasoning", "Standard routing decision.")

        except Exception as e:
            # Fallback heuristic if routing API encounters an issue
            print(f"[SemanticRouter] Fallback triggered ({e})")
            if any(term in lower_q for term in ["revenue", "salary", "compensation", "percent", "fy2024", "fy2023", "market"]):
                chosen_route = RouteType.TABULAR_SQL
                reasoning = "Fallback keyword match: financial metrics detected."
            elif any(term in lower_q for term in ["supplier", "foundry", "tsmc", "hbm", "sk hynix", "partner", "byd", "package"]):
                chosen_route = RouteType.GRAPH_RELATIONS
                reasoning = "Fallback keyword match: entity relationships detected."
            else:
                chosen_route = RouteType.VECTOR_SEARCH
                reasoning = "Fallback default: narrative vector search."

        
         # Assign friendly engine label
        engine_names = {
            RouteType.DIRECT: "Parametric Direct Generator",
            RouteType.TABULAR_SQL: "SQLite Tabular Engine (Text-to-SQL)",
            RouteType.GRAPH_RELATIONS: "Knowledge Graph Engine (Multi-Hop Traversal)",
            RouteType.VECTOR_SEARCH: "Hybrid Vector Engine (Unstructured Text)"
        }

        return RoutingDecision(
            route=chosen_route,
            reasoning=reasoning,
            target_engine=engine_names[chosen_route]
        )
