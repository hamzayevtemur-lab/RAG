"""
Modular RAG — Main Benchmark & Interactive Runner.
Usage:
    cd 03_modular_rag
    python main.py
"""

from config import config
from pipeline import ModularRAGPipeline

def main():
    # Initialize and index
    pipeline = ModularRAGPipeline(config)
    pipeline.index()


    benchmark_queries = [
        # 1. DIRECT ROUTE (Should trigger 0 database lookups)
        "Hello! What kind of questions can you answer about NVIDIA?",
        # 2. TABULAR SQL ROUTE (Should trigger Text-to-SQL on SQLite database)
        "What was the total revenue across all specialized markets in FY2024, and which market grew the fastest?",
        # 3. KNOWLEDGE GRAPH ROUTE (Should trigger multi-hop entity traversal)
        "Which memory suppliers provide HBM3e for the H200, and what packaging bottleneck limits shipping it?",
        # 4. VECTOR SEARCH ROUTE (Should trigger narrative text search)
        "What were the primary risk factors regarding customer concentration and hyperscaler in-house ASICs?",
        # 5. TABULAR SQL ROUTE (Executive Compensation)
        "List all NVIDIA executives who had total compensation greater than $17 million in FY2024."
    ]

    print("\n\n" + "=" * 60)
    print("RUNNING MODULAR RAG BENCHMARK (TESTING ALL 4 ROUTES)")
    print("=" * 60)

    for i, q in enumerate(benchmark_queries, 1):
        print(f"\n[Benchmark Test {i}/{len(benchmark_queries)}]")
        pipeline.query(q, show_details=True)

    # Interactive CLI loop
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE — Test any question (type 'quit' to exit)")
    print("=" * 60)

    while True:
        user_q = input("\n> Enter question: ").strip()
        if user_q.lower() in ("quit", "exit", "q"):
            print("Exiting Modular RAG Explorer. Goodbye!")
            break
        if not user_q:
            continue
        pipeline.query(user_q, show_details=True)

        
if __name__ == "__main__":
    main()