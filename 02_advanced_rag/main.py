"""
Advanced RAG — Main Benchmark & Interactive Runner.
Usage:
    cd 02_advanced_rag
    python main.py
"""

from config import config
from pipeline import AdvancedRAGPipeline

def main():
    # Initialize and index
    pipeline=AdvancedRAGPipeline(config)
    pipeline.index()

    # Benchmark test suite targeting Advanced RAG capabilities
    benchmark_questions = [
        # 1. Exact SKU & Hardware specifications (Tests BM25 + Cross-Encoder precision)
        "How much HBM3e memory does the H200 GPU have, and how does its bandwidth compare to H100?",

        # 2. Regulatory & Geopolitical policy (Tests Query Expansion across acronyms like USG, BIS)
        "What specific export control rules did the U.S. government impose on NVIDIA chips shipped to China in October 2023?",

        # 3. Dense financial tables (Tests Recursive Chunking table boundary preservation)
        "How much revenue did each specialized market generate in FY2024, and what was the percentage change vs FY2023?",

        # 4. Supply chain & capacity commitments (Tests Lexical + Semantic corroboration on large numbers)
        "What are NVIDIA's total outstanding inventory purchase commitments and long-term capacity obligations for TSMC CoWoS packaging?",

        # 5. Out-of-scope question (Tests Strict Zero-Hallucination Defense)
        "What will NVIDIA's stock price be in 2028?"
    ]

    print("\n\n" + "=" * 60)
    print("RUNNING ADVANCED RAG BENCHMARK SUITE")
    print("=" * 60)

    for i, q in enumerate(benchmark_questions, 1):
        print(f"\n[Benchmark Test {i}/{len(benchmark_questions)}]")
        pipeline.query(q, use_expansion=True, show_details=True)

    
    # Interactive CLI loop
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE — Ask anything (type 'quit' or 'exit' to stop)")
    print("=" * 60)

    while True: 
        user_q=input("\n> Enter question: ").strip()
        if user_q.lower() in ("quit", "exit", "q"):
            print("Exiting Advanced RAG Explorer. Goodbye!")
            break
        if not user_q:
            continue
            
        
        pipeline.query(user_q, use_expansion=True, show_details=True)



if __name__ == "__main__":
    main() 

