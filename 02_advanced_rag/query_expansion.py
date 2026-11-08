"""
Pre-Retrieval Query Expansion.
Takes a single user query and generates 3 complementary query
variations using Gemini to overcome vocabulary mismatch and maximize recall.
"""

from google import genai


class MultiQueryExpander:
    """Generates alternative search queries to improve retrieval recall."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
    
    def expand(self, query: str, num_queries: int=3)->list[str]:
        """
        Generate variations of the input query. Always includes the original query.
        """

        prompt = f"""You are an expert AI search engineer. 
Your task is to generate {num_queries - 1} diverse alternative search queries based on the original user question.
Focus on different technical terms, acronyms, and synonyms that might appear in corporate SEC filings, engineering docs, or earnings reports.
Rules:
- Provide exactly {num_queries - 1} alternative queries.
- Put each alternative on its own line.
- Do NOT number them or add any other text or prefixes.
Original Question: {query}
Alternative Queries:"""

        try:
            response=self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            raw_lines=response.text.strip().split("\n")
            alternatives=[
                line.strip() for line in raw_lines
                if line.strip() and not line.strip().startswith(("-", "*", "1.", "2."))
            ]

        except Exception as e:
            print(f"[QueryExpansion] Warning: Failed to expand query ({e}). Using original.")
            alternatives = []

        
        # Always include the original query as the primary query
        all_queries = [query] + alternatives[:num_queries - 1]
        return all_queries
