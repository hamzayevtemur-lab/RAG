"""
Entity-Relationship Knowledge Graph Engine (GraphRAG).

Traverses explicit (Subject, Predicate, Object) graph relationships
across hardware architectures, semiconductor foundries, memory suppliers,
cloud hyperscalers, and export regulatory frameworks.
"""

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GraphTriple:
    subject: str
    predicate: str
    object: str
    context: str



@dataclass
class GraphQueryResult:
    query: str
    matched_entities: list[str]
    triples: list[GraphTriple]
    formatted_paths: str


class KnowledgeGraphEngine:
    """In-memory Knowledge Graph with multi-hop neighborhood traversal."""
    def __init__(self):
        # Adjacency lists for fast bidirectional traversal
        self.out_edges=defaultdict(list)
        self.in_edges=defaultdict(list)
        self._populate_knowledge_graph()

        print(f"[GraphEngine] Initialized Knowledge Graph with {len(self.all_triples)} verified relationships.")    

    def _add_triple(self, sub: str, pred: str, obj: str, context: str=""):
        triple=GraphTriple(subject=sub, predicate=pred, object=obj, context=context)
        self.all_triples.append(triple)
        self.out_edges[sub.lower()].append(triple)
        self.in_edges[obj.lower()].append(triple)

    def _populate_knowledge_graph(self):
        """Build enterprise entity graph extracted from NVIDIA 10-K & Architecture Guide."""
        self.all_triples: list[GraphTriple] = []

        # 1. Hardware Architecture & Silicon Nodes
        self._add_triple("NVIDIA Hopper (H100/H200)", "FABRICATED_ON", "TSMC 4N Process Node", "Custom 4nm process")
        self._add_triple("NVIDIA Blackwell (B200)", "FABRICATED_ON", "TSMC 4NP Process Node", "Dual-die 208B transistors")
        self._add_triple("NVIDIA Grace CPU", "USES_CORE_ARCHITECTURE", "ARM Neoverse V2", "72 ARM cores per CPU")
        self._add_triple("NVIDIA Grace Hopper (GH200)", "INTERCONNECT", "NVLink-C2C", "900 GB/s bidirectional coherent bandwidth")

        # 2. Advanced Packaging & Foundries
        self._add_triple("NVIDIA Hopper (H100/H200)", "PACKAGED_WITH", "TSMC CoWoS", "Chip-on-Wafer-on-Substrate packaging")
        self._add_triple("NVIDIA Blackwell (B200)", "PACKAGED_WITH", "TSMC CoWoS", "Dual-reticle multi-die packaging")
        self._add_triple("TSMC CoWoS", "OUTSTANDING_CAPACITY_COMMITMENT", "$16.05 Billion", "Multi-year capacity reservation guarantee")
        self._add_triple("TSMC CoWoS", "PRIMARY_BOTTLENECK_FACTOR", "Advanced Packaging Capacity", "Revenue constrained by packaging availability")

        # 3. High Bandwidth Memory (HBM) Supply Chain
        self._add_triple("NVIDIA H100 SXM5", "MEMORY_CONFIG", "80GB HBM3 (3,350 GB/s)", "5 active stacks")
        self._add_triple("NVIDIA H200 SXM5", "MEMORY_CONFIG", "141GB HBM3e (4,800 GB/s)", "6 active stacks (76% capacity gain)")
        self._add_triple("NVIDIA B200", "MEMORY_CONFIG", "192GB HBM3e (8,000 GB/s)", "8 active stacks")
        self._add_triple("141GB HBM3e", "PRIMARY_SUPPLIER", "SK Hynix", "Mass producing 8-high HBM3e at 1.2 TB/s per stack")
        self._add_triple("141GB HBM3e", "SECONDARY_SUPPLIERS", "Samsung Electronics & Micron Technology", "Qualification and volume ramp")

        # 4. Regulatory, Geopolitics & China Market
        self._add_triple("NVIDIA Data Center", "REGULATED_BY", "USG Export Control Rules (Oct 2023)", "Restricts high-compute accelerators")
        self._add_triple("USG Export Control Rules (Oct 2023)", "RESTRICTS_EXPORTS_TO", "China, Macau, and Country Groups D1/D4/D5", "Requires explicit export license")
        self._add_triple("USG Export Control Rules (Oct 2023)", "RESTRICTED_CHIPS", "A100, A800, H100, H800, L40, L40S, RTX 4090", "Exceeds performance density thresholds")
        self._add_triple("NVIDIA China Compliant Chip", "COMPLIANT_PRODUCT", "H20 GPU", "Modified density compliant with BIS rules")
        self._add_triple("China Domestic Competition", "PRIMARY_COMPETITOR", "Huawei Ascend 910B", "Fabricated by SMIC on 7nm process")

        # 5. Cloud Hyperscalers & OEM Partnerships
        self._add_triple("NVIDIA DGX Cloud", "HOSTED_ON", "Microsoft Azure", "Dedicated H100 SuperPOD instances")
        self._add_triple("NVIDIA DGX Cloud", "HOSTED_ON", "Oracle Cloud Infrastructure (OCI)", "Largest single deployment: 16,384 GPUs")
        self._add_triple("NVIDIA DGX Cloud", "HOSTED_ON", "Google Cloud Platform", "A3 Mega instance integration")
        self._add_triple("NVIDIA DRIVE Platform", "AUTOMOTIVE_OEM_PARTNER", "Mercedes-Benz", "Next-gen S-Class and EQS architecture")
        self._add_triple("NVIDIA DRIVE Thor", "AUTOMOTIVE_OEM_PARTNER", "BYD", "World's largest EV maker deploying centralized compute")


    def find_subgraph(self, query: str, max_hops: int=2)-> GraphQueryResult:
        """
        Extract relevant entities from the query and traverse their multi-hop graph neighborhood.
        """
        query_lower=query.lower()
        matched_entities=set()


        # Step 1: Detect entity nodes mentioned in query
        all_nodes=set(self.out_edges.keys()) | set(self.in_edges.keys())
        for node in all_nodes:
            # Match if key tokens of the node appear in the query
            node_keywords=[w for w in node.split() if len(w)>2]
            if any(kw in query_lower for kw in node_keywords):
                matched_entities.add(node)

        if not matched_entities:
            # Fallback keyword scan
            for kw in ["h200", "h100", "b200", "tsmc", "cowos", "hbm3e", "sk hynix", "china", "export", "huawei", "azure", "byd"]:
                if kw in query_lower:
                    matched_entities.add(kw)

        
        # Step 2: Multi-Hop Traversal (Breadth-First Search)
        visited_triples: list[GraphTriple]=[]
        seen_triples=set()
        frontier=set(matched_entities)

        for _ in range(max_hops):
            next_frontier=set()
            for entity in frontier:
                # Outgoing edges
                for triple in self.out_edges.get(entity, []):
                    key=(triple.subject, triple.predicate, triple.object)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        visited_triples.append(triple)
                        next_frontier.add(triple.object.lower())
                
                # Incoming edges
                for triple in self.in_edges.get(entity, []):
                    key=(triple.subject, triple.predicate, triple.object)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        visited_triples.append(triple)
                        next_frontier.add(triple.subject.lower())

            
            frontier=next_frontier

        
        # Step 3: Format paths as structured Markdown
        if not visited_triples:
            formatted_paths="No direct knowledge graph relationships found for this query."
        else: 
            lines=[]
            for t in visited_triples:
                extra = f" *(Note: {t.context})*" if t.context else ""
                lines.append(f"- **{t.subject}** ──`[{t.predicate}]`──► **{t.object}**{extra}")

            formatted_paths = "\n".join(lines)

        
        return GraphQueryResult(
            query=query,
            matched_entities=list(matched_entities),
            triples=visited_triples,
            formatted_paths=formatted_paths
        )


