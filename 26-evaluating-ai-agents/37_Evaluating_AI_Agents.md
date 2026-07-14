# 37. Evaluating AI Agents

> **Context:** Section 26. This is a production-grade deep dive into evaluating AI agents and LLM applications. It covers both offline evaluation (testing before deployment) and online evaluation (monitoring in production), addressing the shift from deterministic to non-deterministic validation patterns.

> 💡 **Thesis.** Agent evaluation is not a testing afterthought; it is a production architecture. Traditional unit tests fail for LLMs because outputs are non-deterministic. You must build separate offline (CI/CD) and online (live traffic) evaluation pipelines using LLM-as-Judge, semantic similarity, tool metadata validation, and hallucination detection.

---

## Table of Contents

| # | Section | What You'll Learn |
|---|---------|-------------------|
| 1 | [Why Deterministic Testing Fails for AI](#1-why-deterministic-testing-fails-for-ai) | Why traditional `assert output == expected` breaks with LLMs |
| 2 | [Offline vs. Online Evaluation](#2-offline-vs-online-evaluation) | Two-tier architecture: pre-deployment testing and live traffic monitoring |
| 3 | [Deep Dive: Offline Evaluation](#3-deep-dive-offline-evaluation) | Golden datasets, LLM-as-Judge, semantic scoring, and CI/CD integration |
| 4 | [Deep Dive: Custom Code Evaluators](#4-deep-dive-custom-code-evaluators) | Deterministic guardrails using pure Python logic (regex, JSON schema, type checks) |
| 5 | [Deep Dive: Online Evaluation](#5-deep-dive-online-evaluation) | Reference-free metrics, self-consistency, hallucination detection in production |
| 6 | [Deep Dive: Tool Calling Evaluation](#6-deep-dive-tool-calling-evaluation) | Validating function/tool selection, argument completeness, and type signatures |
| 7 | [Deep Dive: LangSmith Integration](#7-deep-dive-langsmith-integration) | Tracing, debugging, and running evaluations with LangSmith |
| 8 | [Interview Q&A Anchors](#interview-qa-anchors) | Concise production-grade interview answers |
| 9 | [References](#references) | Official docs and further reading |

---

## Key Definitions

| Term | Quick Recall | Full Definition |
|------|--------------|-----------------|
| **Deterministic Testing** | Input → exact same output every time | Classical unit test model where `f(x) = y` always. Fails for LLMs. |
| **Non-Deterministic Testing** | Same input can produce multiple valid outputs | LLM outputs vary even with temperature=0. Must validate semantics, not strings. |
| **Offline Evaluation** | Pre-deployment testing on fixed dataset | Run evaluations during CI/CD using a golden dataset with human-verified references. |
| **Online Evaluation** | Production traffic monitoring | Evaluate live requests in production using reference-free checks (no ground truth). |
| **Golden Dataset** | Canonical test set with references | A fixed collection of input/output pairs, each output verified by humans. |
| **LLM-as-Judge** | Using a strong LLM to grade outputs | Deploy a separate high-tier model (e.g., GPT-4) to evaluate another model's response. |
| **Semantic Similarity** | Comparing meaning, not text | Use embeddings and cosine distance to check if outputs are meaningfully equivalent. |
| **Self-Consistency** | Checking internal model agreement | Sample multiple outputs from the same input; high variance hints at hallucination. |
| **Tool Calling Evaluation** | Validating function invocation metadata | Check tool name, argument keys, and type signatures deterministically. |
| **Hallucination** | Generating plausible but false information | Model confidently invents facts. Detected by self-consistency, RAG grounding, or reference checks. |
| **Reference-Free Evaluation** | Scoring without ground truth | Metrics that work live, when human labels are unavailable. |

---

## 1. Why Deterministic Testing Fails for AI

### The Paradigm Shift

Classical software testing relies on **determinism**: if you pass the same input twice, the system returns the exact same output. This allows rigid assertions:

```python
assert calculate_interest(principal=1000, rate=0.05) == 50.0
```

LLMs and agents operate **non-deterministically**. The same query with identical configuration can produce distinct text outputs, each factually correct and semantically valid:

- **Query:** "What is the capital of France?"
- **Output 1:** "Paris is the capital of France."
- **Output 2:** "France's capital is Paris."
- **Output 3:** "The French capital is Paris."

All three are correct, but string comparison fails:

```python
# This assertion breaks LLM testing
assert output == "Paris is the capital of France."  # May fail even if the model is right
```

### Why It Matters

1. **Test Brittleness** — Tests fail on rephrasing, not actual errors.
2. **False Negatives** — A correct model is marked as broken.
3. **False Positives** — A hallucinating model passes text similarity checks.

Therefore, validation must shift from **rigid text assertions** to **semantic scoring matrices**, **metadata validation**, and **reference-free checks**.

---

## 2. Offline vs. Online Evaluation

AI agent evaluation operates on a **two-tier architecture**:

### Offline Evaluation (Pre-Deployment CI/CD)

**When:** Before shipping code to production.  
**Data:** Fixed, human-verified golden dataset (inputs + references).  
**Evaluators:** LLM-as-Judge, embedding similarity, custom guardrails.  
**Goal:** Ensure quality before users see it.

Example:
- You change the system prompt. Run offline tests against 50 labeled examples to catch regressions.
- A CI/CD pipeline automatically evaluates the new model behavior.
- If scores drop below 0.85, the deployment is blocked.

### Online Evaluation (Production Telemetry)

**When:** After deployment, continuously monitoring live requests.  
**Data:** Streaming user payloads; no human labels available.  
**Evaluators:** Self-consistency, hallucination detection, safety guardrails, cost tracking.  
**Goal:** Catch drift, hallucinations, and cost spikes in production.

Example:
- A user asks a question. The system serves it live and also samples 3 variations with high temperature.
- If the outputs diverge significantly, flag it as a potential hallucination.
- If PII is detected in the output, trigger a security alert.

---

## 3. Deep Dive: Offline Evaluation

Offline evaluation is the foundation. You need a golden dataset and a set of evaluators.

### Building the Golden Dataset

A golden dataset requires:

1. **Inputs** — Real user queries, edge cases, or representative scenarios.
2. **Reference Outputs** — The ideal, human-verified correct answer.

Example dataset:

```python
golden_dataset = [
    {
        "input": {"question": "What is the policy on international remote work?"},
        "reference": "Employees may work internationally up to 30 calendar days per year with prior manager approval."
    },
    {
        "input": {"question": "What happens if an expense report is submitted late?"},
        "reference": "Late expense submissions are deferred to the next bi-weekly payroll processing cycle."
    }
]
```

### LLM-as-Judge Evaluation

Use a strong, independent LLM to grade your model's outputs against the reference.

**Process:**
1. Generate output from your target model.
2. Prompt a judge LLM: "Is this response correct given the reference?"
3. Judge returns a score (0–1 or categorical).

**Prompt Template:**

```
You are a compliance expert. Evaluate whether the following response correctly answers the question using the reference material.

Question: {question}
Reference: {reference}
Actual Response: {actual_output}

Score this response on correctness (0 = completely wrong, 1 = perfect). Explain your reasoning in one sentence.
```

**Pros:**
- Natural language reasoning.
- Catches semantic errors.
- Works for open-ended tasks.

**Cons:**
- Slow (LLM inference cost).
- Judge itself can hallucinate.
- Less deterministic than code-based checks.

### Semantic Similarity (Embedding-Based) Evaluation

Convert both reference and output to embeddings, then compute cosine distance.

**Process:**

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

reference_vec = embedder.embed_query(reference_output)
actual_vec = embedder.embed_query(actual_output)

cosine_similarity = np.dot(reference_vec, actual_vec) / (
    np.linalg.norm(reference_vec) * np.linalg.norm(actual_vec)
)

if cosine_similarity >= 0.85:
    score = 1.0  # Pass
else:
    score = 0.0  # Fail
```

**Pros:**
- Fast and deterministic.
- Robust to paraphrasing.
- Cheap (single embedding inference).

**Cons:**
- Less semantic understanding than LLM-as-Judge.
- Sensitive to threshold tuning.
- May miss factual errors if wording is similar.

### Using LangSmith for Offline Evaluation

LangSmith provides a managed platform for building and running evaluation suites.

**Setup:**

```python
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI

client = Client()

# Create a dataset
dataset = client.create_dataset(dataset_name="ComplianceQA_v1")

client.create_examples(
    inputs=[{"question": "What is the refund policy?"}],
    outputs=[{"reference": "Refunds are allowed within 30 days..."}],
    dataset_id=dataset.id
)

# Define your target function
def compliance_chatbot(inputs):
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    response = model.invoke(inputs["question"])
    return {"answer": response.content}

# Run evaluation
evaluate(
    compliance_chatbot,
    data="ComplianceQA_v1",
    evaluators=["correctness"],  # Built-in evaluator
    experiment_prefix="ci-deployment-check"
)
```

---

## 4. Deep Dive: Custom Code Evaluators

Not every check needs an LLM. Deterministic code-based evaluators are faster and more reliable for structural rules.

### Security & PII Guardrail

Detect sensitive data leakage using regex patterns:

```python
import re

def security_gate(inputs, outputs, reference_outputs=None):
    """Fail if output contains SSN, passwords, or API keys."""
    response = outputs.get("answer", "")
    
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    api_key_pattern = r"(sk_live_|sk_test_)\w+"
    
    if re.search(ssn_pattern, response) or re.search(api_key_pattern, response):
        return {
            "key": "security_check",
            "score": 0.0,
            "comment": "CRITICAL: PII or credentials leaked in output."
        }
    return {
        "key": "security_check",
        "score": 1.0,
        "comment": "No sensitive data detected."
    }
```

### JSON Schema Validation

Ensure structured outputs conform to expected format:

```python
import json

def json_format_check(inputs, outputs, reference_outputs=None):
    """Fail if output is not valid JSON."""
    response = outputs.get("answer", "")
    
    try:
        data = json.loads(response)
        required_keys = {"name", "email", "status"}
        if required_keys.issubset(data.keys()):
            return {
                "key": "json_structure",
                "score": 1.0,
                "comment": "Valid JSON with all required fields."
            }
        else:
            return {
                "key": "json_structure",
                "score": 0.5,
                "comment": f"Missing keys: {required_keys - set(data.keys())}"
            }
    except json.JSONDecodeError:
        return {
            "key": "json_structure",
            "score": 0.0,
            "comment": "Output is not valid JSON."
        }
```

### Tool Calling Metadata Validation

For agents that invoke external tools, validate the metadata exactly:

```python
def tool_metadata_check(inputs, outputs, reference_outputs):
    """Verify tool name, arguments, and types."""
    generated_calls = outputs.get("tool_calls", [])
    expected_call = reference_outputs.get("reference_tool_call")
    
    if not generated_calls:
        return {"key": "tool_check", "score": 0.0, "comment": "No tools called."}
    
    actual_call = generated_calls[0]
    
    # Check 1: Tool name matches
    if actual_call["name"] != expected_call["name"]:
        return {
            "key": "tool_check",
            "score": 0.0,
            "comment": f"Wrong tool: got {actual_call['name']}, expected {expected_call['name']}."
        }
    
    # Check 2: Exact argument match
    if actual_call["args"] == expected_call["args"]:
        return {"key": "tool_check", "score": 1.0, "comment": "Perfect tool invocation."}
    
    # Check 3: Argument keys match but values differ (partial credit)
    if set(actual_call["args"].keys()) == set(expected_call["args"].keys()):
        return {
            "key": "tool_check",
            "score": 0.5,
            "comment": "Tool and keys correct; argument values differ."
        }
    
    return {"key": "tool_check", "score": 0.0, "comment": "Tool argument mismatch."}
```

### Combining Evaluators in LangSmith

```python
evaluate(
    compliance_chatbot,
    data="ComplianceQA_v1",
    evaluators=[
        "correctness",      # LLM-as-Judge
        security_gate,      # Custom PII check
        json_format_check,  # Custom format check
        tool_metadata_check # Tool validation
    ]
)
```

---

## 5. Deep Dive: Online Evaluation

Online evaluation happens in production with no human labels. You must rely on **reference-free metrics**.

### Self-Consistency for Hallucination Detection

If a model truly knows a fact, it will reproduce the same core meaning across high-temperature samples. Hallucinations diverge widely.

**Algorithm:**

1. Generate the primary output (temperature=0 for consistency).
2. Sample 3–5 variations of the same input with high temperature (e.g., 0.8).
3. Embed all outputs and compute pairwise cosine distances.
4. If average distance is high, flag as potential hallucination.

**Code:**

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import numpy as np

def hallucination_detector(user_question, primary_output):
    """Detect hallucination using self-consistency."""
    model = ChatOpenAI(model="gpt-4o", temperature=0.8)
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Sample variations
    samples = [model.invoke(user_question).content for _ in range(3)]
    
    # Embed primary and samples
    primary_vec = embedder.embed_query(primary_output)
    sample_vecs = embedder.embed_documents(samples)
    
    # Compute distances
    distances = []
    for vec in sample_vecs:
        cos_dist = 1.0 - np.dot(primary_vec, vec) / (
            np.linalg.norm(primary_vec) * np.linalg.norm(vec)
        )
        distances.append(cos_dist)
    
    avg_variance = sum(distances) / len(distances)
    
    if avg_variance > 0.25:
        return {
            "key": "hallucination_risk",
            "score": 0.0,
            "comment": f"High divergence: {avg_variance:.2f}. Likely hallucination."
        }
    return {
        "key": "hallucination_risk",
        "score": 1.0,
        "comment": f"Stable output: {avg_variance:.2f}. Low hallucination risk."
    }
```

### Cost Tracking

Monitor token consumption and cost per request in production:

```python
def cost_monitor(inputs, outputs, metadata):
    """Track inference costs."""
    tokens_used = metadata.get("usage", {}).get("total_tokens", 0)
    cost_per_token = 0.00003  # Example: GPT-4o pricing
    request_cost = tokens_used * cost_per_token
    
    if request_cost > 0.10:  # Flag if exceeds $0.10
        return {
            "key": "cost_anomaly",
            "score": 0.0,
            "comment": f"High cost: ${request_cost:.4f} for {tokens_used} tokens."
        }
    return {
        "key": "cost_monitoring",
        "score": 1.0,
        "comment": f"Cost normal: ${request_cost:.4f}."
    }
```

---

## 6. Deep Dive: Tool Calling Evaluation

Agents often invoke external tools (database queries, API calls, etc.). Evaluation must validate the metadata.

### The Three Fault Planes

| Fault Type | Description | Detection |
|---|---|---|
| **Selection Error** | Wrong tool invoked | Tool name does not match expected |
| **Completeness Error** | Missing required arguments | Argument keys are a subset of expected |
| **Type Error** | Wrong argument type | Argument is a string instead of int |

### Full Tool Validation

```python
def tool_call_validator(inputs, outputs, reference_outputs):
    """Comprehensive tool calling evaluation."""
    generated_calls = outputs.get("tool_calls", [])
    expected = reference_outputs.get("reference_tool_call")
    
    if not generated_calls:
        return {"key": "tool_validation", "score": 0.0, "comment": "No tools called."}
    
    actual = generated_calls[0]
    issues = []
    
    # Check 1: Tool name
    if actual["name"] != expected["name"]:
        issues.append(f"Tool: got '{actual['name']}', expected '{expected['name']}'.")
    
    # Check 2: Required keys
    missing_keys = set(expected["args"].keys()) - set(actual["args"].keys())
    if missing_keys:
        issues.append(f"Missing arguments: {missing_keys}.")
    
    # Check 3: Type mismatches
    for key, expected_type in expected.get("arg_types", {}).items():
        if key in actual["args"]:
            actual_type = type(actual["args"][key]).__name__
            if actual_type != expected_type:
                issues.append(f"Argument '{key}': got {actual_type}, expected {expected_type}.")
    
    if not issues:
        return {"key": "tool_validation", "score": 1.0, "comment": "Perfect tool invocation."}
    
    return {
        "key": "tool_validation",
        "score": max(0.0, 1.0 - (len(issues) * 0.25)),
        "comment": " | ".join(issues)
    }
```

---

## 7. Deep Dive: LangSmith Integration

LangSmith provides a unified platform for tracing, debugging, and evaluating agents.

### Setting Up Tracing

```python
from langsmith import Client
from langchain_core.traceable import traceable
from langchain_openai import ChatOpenAI

client = Client()

@traceable
def my_agent(question):
    model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    response = model.invoke(question)
    return response.content

# Calls are automatically logged
result = my_agent("What is the capital of France?")
```

All calls are visible in the LangSmith dashboard, including:
- Full input/output traces.
- Token usage.
- Latency.
- Errors.

### Running Evaluations with LangSmith

```python
from langsmith import evaluate

evaluate(
    my_agent,
    data="MyDataset",
    evaluators=[
        "correctness",
        security_gate,
        tool_metadata_check
    ],
    experiment_prefix="production-checkpoint"
)
```

Results are logged to the LangSmith dashboard where you can:
- Compare experiment runs.
- Drill into failing examples.
- Export results for reporting.

---

## Operational Summary

| Evaluation Layer | Timing | Focus | Primary Metric | Tool |
|---|---|---|---|---|
| **Offline LLM-as-Judge** | CI/CD | Semantic correctness | Score 0–1 | LangSmith |
| **Offline Embedding Similarity** | CI/CD | Paraphrase robustness | Cosine > 0.85 | LangSmith |
| **Offline Custom Guards** | CI/CD | PII, JSON, schemas | Binary pass/fail | LangSmith + Custom Code |
| **Online Self-Consistency** | Production | Hallucination risk | Variance < 0.25 | Custom logging |
| **Online Cost Tracking** | Production | Infrastructure spend | $/request threshold | Custom monitoring |
| **Online Tool Validation** | Production | Tool invocation correctness | Binary pass/fail | Custom code |

---

## Interview Q&A Anchors

**Q: Why does traditional unit testing fail for LLMs?**
> **A:** LLMs are non-deterministic. The same input can produce multiple valid outputs with different wording. Traditional `assert output == expected` breaks because it checks string equality, not semantic correctness. Instead, you need semantic scoring matrices and reference-free checks.

**Q: What is the difference between offline and online evaluation?**
> **A:** Offline evaluation runs before deployment using a human-verified golden dataset with known references. Online evaluation monitors live traffic in production without ground truth, using reference-free metrics like self-consistency and hallucination detection.

**Q: How does LLM-as-Judge work?**
> **A:** You deploy a strong, independent LLM (e.g., GPT-4) to grade your model's outputs against a reference answer using natural language reasoning. The judge returns a score 0–1. Pros: semantically rich. Cons: slow, expensive, judge can hallucinate.

**Q: When should you use semantic similarity instead of LLM-as-Judge?**
> **A:** Use embedding-based similarity for speed and cost when you need to handle paraphrasing but don't require deep semantic understanding. It's 10x faster and cheaper. Use LLM-as-Judge for complex tasks where natural language reasoning is essential.

**Q: How do you detect hallucinations in production without ground truth?**
> **A:** Use self-consistency: sample multiple outputs from the same input with high temperature. If outputs diverge significantly (high cosine distance variance), it suggests the model is guessing. Variance > 0.25 typically indicates hallucination risk.

**Q: What are the three fault planes in tool calling evaluation?**
> **A:** Selection (wrong tool name), Completeness (missing required arguments), and Type (argument type mismatch). Deterministic code-based checks validate all three without needing an LLM.

**Q: How do you prevent PII leakage in agent outputs?**
> **A:** Use custom regex evaluators to scan outputs for SSN patterns, API keys, and credentials before returning to the user. This is a deterministic guardrail that runs on 100% of production traffic.

**Q: How does LangSmith simplify evaluation?**
> **A:** LangSmith provides a managed platform for creating golden datasets, running offline evaluations against multiple evaluators, and tracking production traces and costs. It eliminates the need to build custom evaluation infrastructure.

---

## References

- LangSmith documentation: https://docs.smith.langchain.com/
- LangChain evaluation guide: https://python.langchain.com/docs/guides/evaluation/
- OpenAI evaluation best practices: https://platform.openai.com/docs/guides/evals
- Semantic similarity and embeddings: https://www.sbert.net/
- Self-consistency prompting: https://arxiv.org/abs/2203.11171
- RAG grounding for hallucination reduction: https://arxiv.org/abs/2307.03172
