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
Anoop is a Principal/Senior Software Engineer with 12+ years of experience designing scalable, reliable, and maintainable software systems. Passionate about leveraging agile methodologies and cloud platforms to deliver innovative solutions to complex challenges.
🔹 Professional Expertise:
Front-End Development: Proficient in TypeScript, React, Angular, CSS, and various CSS frameworks like Bootstrap with a strong ability to create dynamic, responsive, and visually appealing user interfaces.
Back-End Development: Skilled in C#, Java, Golang, Python, and Node.js, with extensive experience in building robust and efficient server-side applications using technologies like ASP.NET MVC, .NET Core, Web API/REST API, and gRPC.
Distributed Systems: Experienced in designing and implementing distributed systems using technologies like Node.js, gRPC, ASP.NET Core, Kubernetes, Docker, Apache Kafka, RabbitMQ, and Redis. Proficient in building scalable, reliable, high-performance systems that leverage microservices architecture, containerization, orchestration, message queues, and distributed databases to handle large-scale, complex applications.
Mobile Development: Experienced in Android development, delivering high-quality mobile applications.
Cloud Development: AWS Certified, with hands-on experience in Azure and distributed systems, leveraging cloud platforms to enhance scalability and performance.
Event-Driven Programming: Expertise in designing and implementing event-driven architectures, utilizing technologies such as AWS SQS, AWS SNS, and AWS Lambda to build highly responsive and scalable systems.
Databases: Proficient in SQL and NoSQL databases with expertise in ensuring efficient data management and retrieval. Skilled in using ORM frameworks such as EF & EFCore and Dapper to streamline database operations and enhance productivity
Caching Technologies: Experienced with Distributed caching solutions such as Redis and Memcached to improve application performance and scalability.
DevOps: Expertise in continuous integration and deployment, ensuring smooth and efficient delivery pipelines.
🎓 Academic Background:
Master's degree in Computer Science from the University of New Orleans.
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

```

## 3. Technical Deep Dive

### A. `PromptTemplate` vs. `ChatPromptTemplate`

This is the most critical distinction in modern LLM development.

| Feature | `PromptTemplate` | `ChatPromptTemplate` |
| --- | --- | --- |
| **Input Type** | A single raw string. | A list of `Message` objects (System, Human, AI). |
| **Structure** | Unstructured text. | Structured roles. |
| **Use Case** | Simple completion tasks; legacy models. | **Production Standard.** Chatbots, complex reasoning, agents. |

* **`ChatPromptTemplate` Mechanics:**
* **System Message:** Sets the behavior/persona (e.g., "You are a helpful assistant"). This instruction persists even as the conversation grows.
* **Human Message:** The user's input (e.g., the `{information}` variable).
* **AI Message:** The model's response.



### B. The `ChatOpenAI` Model Wrapper

* **Definition:** A class that standardizes the API connection to OpenAI.
* **Key Behavior:** Unlike older models that accepted a string and returned a string, `ChatOpenAI` expects a list of messages and returns an `AIMessage` object.
* **`temperature=0`:** A hyperparameter that controls randomness. Setting it to 0 forces the model to select the highest-probability token at every step, making the output **deterministic** (repeatable).

### C. LCEL (LangChain Expression Language)

* **Definition:** A declarative coding style for composing chains using the pipe operator (`|`).
* **Syntax:** `chain = prompt | model | output_parser`
* **Data Flow:**
1. **Input Dictionary:** `{"information": "..."}`
2. **Prompt:** Receives dictionary  Returns `PromptValue` (formatted string or messages).
3. **Model:** Receives `PromptValue`  Returns `AIMessage`.


* **Why it matters:** It abstracts the data passing. You don't need to manually take the string from step 1 and pass it to function 2. The `|` operator handles the handshake automatically.

### D. `AIMessage`

* **Definition:** The structured response object from a Chat Model.
* **Attributes:**
* `.content`: The actual text of the response.
* `.response_metadata`: Usage statistics (input tokens, output tokens, total cost).



---

## 4. Interview Q&A Anchors

**Q: When should I use `ChatPromptTemplate` over `PromptTemplate`?**

> **A:** You should almost always use `ChatPromptTemplate` when working with modern Chat Models. These models are trained to understand the distinction between a "System Instruction" and a "User Message," making prompts more robust and secure.

**Q: Explain the data flow in the LCEL chain `prompt | model`.**

> **A:** The `.invoke()` method passes a dictionary to the `prompt`. The `prompt` formats this into a `String` (or `List[Message]`) and passes it to the `model`. The `model` processes it and returns an `AIMessage` object.

**Q: Why is `temperature=0` important for this specific task (summarization)?**

> **A:** Summarization is a factual extraction task. We want the model to be faithful to the source text. Higher temperatures introduce randomness, which increases the risk of "hallucination." Temperature 0 minimizes this risk.
