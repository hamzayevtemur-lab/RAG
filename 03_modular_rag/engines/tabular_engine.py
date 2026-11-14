"""
Structured Tabular & SQL Engine.

Initializes an in-memory SQLite database populated with exact
financial and governance tables from the NVIDIA SEC 10-K report.
Converts analytical questions into deterministic SQL queries.
"""
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from google import genai


@dataclass
class TabularQueryResult:
    query: str
    sql: str
    columns: list[str]
    rows: list[tuple]
    formatted_markdown: str




class TabularEngine:
    """Connects to nvidia_financials.db and executes Text-to-SQL."""

    def __init__(self, db_path: Path, api_key: str, model_name: str="gemini-3.1-flash-lite"):
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found at: {db_path}")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.schema = self._fetch_schema()
        print(f"[TabularEngine] Connected to database: {db_path.name}")


    def _fetch_schema(self)->str:
        """Dynamically read CREATE TABLE definitions from SQLite."""
        cursor=self.conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables=cursor.fetchall()
        return "\n\n".join(t[0] for t in tables if t[0])


    def execute_query(self, user_question:str)-> TabularQueryResult:
        """Convert natural language question -> SQL -> Execute on SQLite."""
        prompt = f"""You are an expert SQL compiler for SQLite.
Given the database schema below and the user's question, write a single valid, read-only SQL query.

<schema>
{self.schema}
</schema>

RULES:
- Return ONLY the raw SQL query. Do not wrap in ```sql markdown code fences.
- Use only SELECT statements.
- Use exact table and column names from the schema.
User Question: {user_question}
SQL Query:"""

        response=self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        sql_query=response.text.strip().replace("```sql", "").replace("```", "").strip()


        # Execute query on SQLite with safety
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        except Exception as err:
            return TabularQueryResult(
                query=user_question,
                sql=sql_query,
                columns=["Error"],
                rows=[(str(err),)],
                formatted_markdown=f"**SQL Execution Error:** `{err}`",
            )

        # Format output as clean Markdown table
        if not rows:
            formatted_md = "Query returned 0 rows."
        else:
            header = "| " + " | ".join(columns) + " |"
            sep = "| " + " | ".join(["---"] * len(columns)) + " |"
            data_rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in rows]
            formatted_md = "\n".join([header, sep] + data_rows)

        return TabularQueryResult(
            query=user_question,
            sql=sql_query,
            columns=columns,
            rows=rows,
            formatted_markdown=formatted_md,
        )
