"""
Modular RAG Pipeline — Master Orchestrator.
Flow:
    User Question
         │
         ▼
    [Semantic Router] ──► Classifies intent into 1 of 4 routes
         │
         ├─────────────────┬──────────────────┬─────────────────┐
         ▼                 ▼                  ▼                 ▼
     [DIRECT]        [TABULAR_SQL]     [GRAPH_RELATIONS] [VECTOR_SEARCH]
     (0 retrieval)   (SQLite engine)   (Knowledge Graph) (Dense vectors)
         │                 │                  │                 │
         └─────────────────┴─────────┬────────┴─────────────────┘
                                     ▼
                            [RAG Synthesizer]
                                     ▼
                               Final Answer
"""


from dataclasses import dataclass
from config import ModularRAGConfig
from router import SemanticRouter, RoutingDecision, RouteType
from synthesizer import ModularSynthesizer, SynthesisResult
from engines.tabular_engine import TabularEngine, TabularQueryResult
from engines.graph_engine import KnowledgeGraphEngine, GraphQueryResult
from engines.vector_engine import VectorEngine, VectorQueryResult


@dataclass
class ModularQueryResult:
    question: str
    routing: RoutingDecision
    evidence_markdown: str
    answer: str


class ModularRAGPipeline:
    """End-to-end Modular RAG pipeline."""
    def __init__(self, config: ModularRAGConfig):
        self.config=config

        print("\n" + "=" * 60)
        print("INITIALIZING MODULAR RAG SYSTEM")
        print("=" * 60)

        # 1. Initialize specialized engines
        self.tabular_engine=TabularEngine(
            db_path=config.db_path,
            api_key=config.gemini_api_key,
            model_name=config.llm_model
        )
        self.graph_engine = KnowledgeGraphEngine()
        self.vector_engine = VectorEngine(model_name=config.embedding_model)

        # 2. Initialize router & synthesizer
        self.router=SemanticRouter(api_key=config.gemini_api_key, model_name=config.llm_model)
        self.synthesizer=ModularSynthesizer(
            api_key=config.gemini_api_key, model_name=config.llm_model
        )

    def index(self):
        """Index narrative documents for the Vector Engine."""
        print("\n" + "=" * 60)
        print("INDEXING NARRATIVE TEXT (MODULAR VECTOR ENGINE)")
        print("=" * 60)

        self.vector_engine.index_corpus(data_dir=self.config.data_dir)

        print("[ModularPipeline] All 3 retrieval engines ready.")
        print("=" * 60)


    def query(self, question: str, show_details: bool=True)-> ModularQueryResult:
        """Execute full Modular RAG routing and synthesis."""
        print(f"\n{'─' * 60}")
        print(f"[Question] {question}")
        print(f"{'─' * 60}")

        # Step 1: Semantic Intent Routing
        decision: RoutingDecision=self.router.route_query(question)
        if show_details:
            print(f"[Router] Selected Route : {decision.route.value}")
            print(f"[Router] Target Engine  : {decision.target_engine}")
            print(f"[Router] Reasoning      : {decision.reasoning}")


        # Step 2: Route-specific execution
        engine_output = None
        if decision.route == RouteType.DIRECT:
            engine_output = None

        elif decision.route == RouteType.TABULAR_SQL:
            engine_output: TabularQueryResult = self.tabular_engine.execute_query(question)
            if show_details:
                print(f"[SQL Executed] {engine_output.sql}")

        elif decision.route == RouteType.GRAPH_RELATIONS:
            engine_output: GraphQueryResult = self.graph_engine.find_subgraph(
                question,
                max_hops=self.config.graph_max_hops
            )
            if show_details:
                print(f"[Graph] Matched Entities: {engine_output.matched_entities}")
                print(f"[Graph] Discovered Paths: {len(engine_output.triples)} triples traversed.")

        elif decision.route == RouteType.VECTOR_SEARCH:
            engine_output: VectorQueryResult = self.vector_engine.search(
                question,
                top_k=self.config.vector_top_k
            )
            if show_details:
                print(f"[Vector] Retrieved {len(engine_output.top_chunks)} narrative chunks.")

        # Step 3: Grounded Synthesis
        synth: SynthesisResult=self.synthesizer.synthesize(
            question=question,
            route_type=decision.route,
            engine_output=engine_output
        )

        if show_details:
            print(f"\n[Answer]\n{synth.answer}\n")

        return ModularQueryResult(
            question=question,
            routing=decision,
            evidence_markdown=synth.evidence_markdown,
            answer=synth.answer
        )