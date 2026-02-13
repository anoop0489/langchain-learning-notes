# 01. LangChain Basics: PromptTemplates & LCEL

## 1. Core Concept: The Prompt-Model Chain
This module covers the fundamental building block of LangChain: creating a linear pipeline that processes user input and generates a response using an LLM.
We use **Declarative Orchestration (LCEL)** to link a **Prompt Template** (the instructions) with a **Chat Model** (the intelligence).

## 2. Code Implementation
*This script demonstrates two approaches: Standard `PromptTemplate` (String-based) and `ChatPromptTemplate` (Message-based).*

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Load Environment Variables
load_dotenv()

# 2. The Context (Raw Data)
information = """
Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, known for his leadership of Tesla, SpaceX, X (formerly Twitter)...
[...full text truncated...]
"""

# --- APPROACH 1: Standard PromptTemplate (String Manipulation) ---
# Use this for simple, non-conversational tasks or older completion models.
summary_template = """
given the information {information} about a person I want you to create:
1. A short summary
2. two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"], 
    template=summary_template
)

# --- APPROACH 2: ChatPromptTemplate (Message-Based) ---
# Use this for Chat Models (GPT-4, Claude, Gemini). It structures input into Roles.
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You are given information about a person and you need to create a summary and two interesting facts."),
    ("human", "{information}")
])

# 3. The Model (LLM Wrapper)
# temperature=0 ensures deterministic output (critical for factual extraction).
llm = ChatOpenAI(temperature=0, model="gpt-4o")

# 4. The Chain (LCEL Syntax)
# We pipe the specific template into the model.
chain = chat_template | llm

# 5. Execution
response = chain.invoke(input={"information": information})

# 6. Output
print(response.content)
