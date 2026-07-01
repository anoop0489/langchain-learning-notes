# ---------------------------------------------------------------------------
# main.py - Agentic RAG Entry Point
# ---------------------------------------------------------------------------
# Runs the full Agentic RAG graph: routes the question, retrieves from
# vectorstore or web, grades documents, generates an answer, checks for
# hallucinations, and verifies the answer addresses the question.
#
# Prerequisites:
#   - Run ingestion first: cd 16-agentic-rag/src && uv run python ingestion.py
#   - Set OPENAI_API_KEY and TAVILY_API_KEY in .env
#
# Run:
#   cd 16-agentic-rag/src
#   uv run python main.py
# ---------------------------------------------------------------------------

import os
import sys

import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from graph.graph import app


def check_prerequisites():
    required = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        print("   Add them to your .env file")
        sys.exit(1)


if __name__ == "__main__":
    check_prerequisites()

    print("=" * 60)
    print("AGENTIC RAG - Adaptive Retrieval with Self-Correction")
    print("=" * 60)

    print("\nGraph Mermaid Diagram:")
    print(app.get_graph().draw_mermaid())

    print("\n" + "-" * 60)
    print("Query: 'agent memory'")
    print("-" * 60)
    result = app.invoke(input={"question": "agent memory"})
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["generation"])
