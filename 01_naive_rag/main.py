"""
Naive RAG — Main Entry Point.

Usage:
    cd 01_naive_rag
    python main.py

First run downloads the embedding model (~80 MB).
Subsequent runs load from cache instantly.
"""

from config import config
from pipeline import NaiveRAGPipeline


def main():
    # ── Initialize and index ───────────────────────────
    pipeline = NaiveRAGPipeline(config)
    pipeline.index()

    # ── Test questions ─────────────────────────────────
    # These are designed to test different retrieval challenges:
    test_questions = [
        # Simple factual (answer in one chunk)
        "What was NVIDIA's total revenue in fiscal year 2024?",

        # Requires specific table data
        "How much revenue did the Data Center segment generate in FY2024 vs FY2023?",

        # Cross-document (architecture guide + 10-K)
        "How much HBM3e memory does the H200 GPU have and how does it compare to H100?",

        # Earnings call specific
        "What revenue guidance did NVIDIA give for Q1 FY2025?",

        # Multi-hop (needs competition file + 10-K)
        "What is NVIDIA's market share in data center AI accelerators?",

        # Unanswerable (tests hallucination defense)
        "What is the price of NVIDIA stock today?",
    ]

    print("\n\n" + "=" * 60)
    print("QUERYING PHASE — Testing 6 Questions")
    print("=" * 60)

    for question in test_questions:
        pipeline.query(question)
        print("\n")

    # ── Interactive mode ───────────────────────────────
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE — Type your questions (type 'quit' to exit)")
    print("=" * 60)

    while True:
        question = input("\n> Your question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        pipeline.query(question)


if __name__=="__main__":
    main()



