"""
LLM Client — Gemini API for Grounded Generation.

Takes retrieved chunks + user question → constructs a grounded
prompt → calls Gemini → returns the answer.

The prompt design forces the LLM to:
1. ONLY use information from the provided context chunks
2. Say "I don't have enough information" if the answer isn't in the chunks
3. Cite which chunk(s) the answer came from
"""
from google import genai


class LLMClient:
    """Lightweight wrapper around Gemini API for RAG generation."""
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the Gemini client.
        Args:
            api_key:    Your Gemini API key from https://aistudio.google.com
            model_name: Model to use (e.g., "gemini-2.5-flash").
        """
        self.client=genai.Client(api_key=api_key)
        self.model_name=model_name
        print(f"[LLM] Initialized - model: {model_name}")

    def generate(self, query: str, context_chunks: list[dict])-> str:
        """
        Generate an answer grounded in the retrieved chunks.

        Args:
            query:          The user's question.
            context_chunks: List of dicts with keys:
                            - "text": chunk content
                            - "source": source filename
                            - "score": similarity score

        Returns:
            The generated answer string.
        """
        
        # Build the context block from retrieved chunks
        context_block=""
        for i, chunk in enumerate(context_chunks, 1):
            context_block+=(
                f"\n<chunk id=\"{i}\" source=\"{chunk['source']}\" "
                f"score=\"{chunk['score']:.3f}\">\n"
                f"{chunk['text']}\n"
                f"</chunk>\n"
            )

        # Construct the grounded prompt
        prompt = f"""You are a precise analyst. Answer the user's question using ONLY 
the information provided in the context chunks below.

RULES:
- Use ONLY facts from the provided chunks. Do NOT use outside knowledge.
- If the chunks do not contain enough information, say: "The provided context does not contain enough information to answer this question."
- Cite the chunk number(s) you used, e.g., [Chunk 1], [Chunk 3].
- Be specific with numbers, dates, and percentages — do not round or approximate.

<context>
{context_block}
</context>

<question>
{query}
</question>

Answer:"""

        # Call Gemini API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text