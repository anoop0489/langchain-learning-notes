# ---------------------------------------------------------------------------
# generation.py - RAG generation chain
# ---------------------------------------------------------------------------
# The actual answer generation step. Uses the standard rlm/rag-prompt from
# LangChain Hub which formats documents + question into a clean RAG prompt.
# This is the ONLY chain that returns freeform text (not structured output).
# ---------------------------------------------------------------------------

import os
import sys

import truststore
truststore.inject_into_ssl()
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "agentic-rag"

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
