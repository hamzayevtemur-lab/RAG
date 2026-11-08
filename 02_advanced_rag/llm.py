"""
Advanced LLM Generator.
Takes cross-encoder re-ranked chunks + original user query,
constructs a grounded prompt with explicit provenance metadata,
and generates the final cited answer using Gemini.
"""

from google import genai
from reranker import RerankedResult


class AdvancedLLMClient:
    """Gemini client for grounded generation in Advanced RAG."""

    def __init__(self, api_key: str, model_name: str="gemini-3.1-flash-lite"):
        self.client=genai.Client(api_key=api_key)
        self.model_name=model_name

        print(f"[LLM] Initialized Advanced Generator with model: {model_name}")

    def generate(self, query: str, reranked_chunks: list[RerankedResult])->str:
        """
        Generate an answer strictly grounded in the re-ranked chunks.
        Args:
            query:           The user's original question.
            reranked_chunks: Top-K candidates ordered by Cross-Encoder score.
        Returns:
            Grounded response string with bracketed citations.
        """
        # Build XML context with cross-encoder relevance scores
        context_blocks=[]
        for r in reranked_chunks:
            dense_str=f"#{r.dense_rank}" if r.dense_rank is not None else "N/A"
            sparse_str=f"#{r.sparse_rank}" if r.sparse_rank is not None else "N/A"

            block = (
                f'<chunk id="{r.final_rank}" '
                f'source="{r.chunk.source_file}" '
                f'cross_encoder_score="{r.rerank_score:.4f}" '
                f'dense_rank="{dense_str}" '
                f'bm25_rank="{sparse_str}">\n'
                f"{r.chunk.text}\n"
                f"</chunk>"
            )

            context_blocks.append(block)

        full_context = "\n\n".join(context_blocks)

        prompt = f"""You are a senior technical and financial research analyst.
Answer the user's question using ONLY the provided verified context chunks below.

RULES:
1. Base your answer STRICTLY on the facts directly mentioned in the context chunks. Do not extrapolate or assume.
2. If the context does not contain sufficient information to fully answer the question, clearly state: "The provided documents do not contain enough information to answer this question."
3. Every factual claim, figure, date, and metric MUST be cited with its corresponding chunk ID in brackets, e.g., [Chunk 1] or [Chunk 2].
4. When reporting numbers, percentages, or hardware specifications, maintain exact numerical precision as stated in the text.

<verified_context>
{full_context}
</verified_context>

<user_question>
{query}
</user_question>

Grounded Answer:"""

        response=self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text.strip()

        