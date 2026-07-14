# Evaluating AI Agents — Implementation Guide

> **What You'll Build:** A production-grade offline evaluation pipeline using LangSmith, custom evaluators, and online monitoring logic. You'll evaluate a compliance chatbot against a golden dataset and integrate safety guardrails.

---

## Project Structure

```
26-evaluating-ai-agents/
├── 37_Evaluating_AI_Agents.md           # Theory & concepts
├── 38_Evaluating_AI_Agents_Implementation.md  # This file
├── src/
│   ├── main.py                          # Core offline evaluation pipeline
│   ├── custom_evaluators.py             # Reusable guardrail logic
│   ├── ingestion.py                     # Golden dataset creation
│   ├── test_online_monitoring.py        # Self-consistency & hallucination detection
│   └── test_tool_validation.py          # Tool calling evaluation
└── assets/
    └── sample_golden_dataset.json       # Example dataset for testing
```

---

## Dependencies

Install all required packages:

```bash
uv add langsmith langchain langchain-openai langchain-core python-dotenv numpy
```

**Key packages:**
- `langsmith` — Evaluation platform and dataset management
- `langchain-openai` — OpenAI models and embeddings
- `numpy` — Vector math for semantic similarity

---

## Environment Variables

Create a `.env` file in the repository root:

```env
# OpenAI API
OPENAI_API_KEY=sk_...your_key_here

# LangSmith API (for tracing and evaluation)
LANGSMITH_API_KEY=ls_...your_key_here
LANGSMITH_PROJECT=evaluating-ai-agents

# Optional: Local LLM via Ollama (for cost-saving in development)
OLLAMA_MODEL=qwen:1.7b
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Implementation Walkthrough

### Step 1: Core Offline Evaluation Pipeline (`main.py`)

This script sets up a golden dataset and runs offline evaluations using LangSmith.

```python
"""
Offline Evaluation Pipeline for Compliance Chatbot

This script:
1. Creates a golden dataset of compliance Q&A pairs
2. Defines a target compliance chatbot function
3. Runs offline evaluations using LLM-as-Judge and custom guardrails
4. Logs results to LangSmith for dashboard inspection

Prerequisites:
- OPENAI_API_KEY in .env
- LANGSMITH_API_KEY in .env

Run: uv run main.py
"""

import os
import truststore
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI

# Inject corporate proxy SSL certificates
truststore.inject_into_ssl()

load_dotenv()

def check_prerequisites():
    """Validate required environment variables."""
    required_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")
    print("✅ Prerequisites validated")

def create_or_get_golden_dataset():
    """Initialize the golden dataset in LangSmith."""
    client = Client()
    dataset_name = "ComplianceQA_v1"
    
    # Check if dataset already exists
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✅ Using existing dataset: {dataset_name}")
        return dataset
    except Exception:
        print(f"📝 Creating new dataset: {dataset_name}")
    
    # Create dataset with sample examples
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Production reference dataset for compliance chatbot evaluation"
    )
    
    examples = [
        {
            "input": {"question": "What is the policy on international remote work?"},
            "output": {
                "answer": "Employees may work internationally up to 30 calendar days per year with prior manager approval.",
                "reference": "Employees may work internationally up to 30 calendar days per year with prior manager approval."
            }
        },
        {
            "input": {"question": "What happens if an expense report is submitted late?"},
            "output": {
                "answer": "Late expense submissions are deferred to the next bi-weekly payroll processing cycle.",
                "reference": "Late expense submissions are deferred to the next bi-weekly payroll processing cycle."
            }
        },
        {
            "input": {"question": "What is the refund policy?"},
            "output": {
                "answer": "Refunds are allowed within 30 days of purchase with original receipt.",
                "reference": "Refunds are allowed within 30 days of purchase with original receipt."
            }
        }
    ]
    
    # Add examples to dataset
    for example in examples:
        client.create_example(
            inputs=example["input"],
            outputs=example["output"],
            dataset_id=dataset.id
        )
    
    print(f"✅ Dataset created with {len(examples)} examples")
    return dataset

def compliance_chatbot(inputs: dict) -> dict:
    """Target application: compliance chatbot."""
    question = inputs.get("question", "")
    
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,  # Deterministic for offline evaluation
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    system_prompt = """You are a compliance expert. Answer policy questions concisely and factually.
    Base your answers only on company policy. If you don't know, say so."""
    
    response = model.invoke(system_prompt + f"\n\nQuestion: {question}")
    return {"answer": response.content}

def run_offline_evaluation():
    """Execute the offline evaluation pipeline."""
    print("\n🚀 Starting Offline Evaluation Pipeline...\n")
    
    check_prerequisites()
    create_or_get_golden_dataset()
    
    print("📊 Running evaluations with LLM-as-Judge...\n")
    
    # Import custom evaluators
    from custom_evaluators import security_gate, json_format_check
    
    # Run evaluation using LangSmith
    evaluate(
        compliance_chatbot,
        data="ComplianceQA_v1",
        evaluators=[
            "correctness",          # Built-in LLM-as-Judge
            security_gate,          # Custom PII check
            json_format_check       # Custom format check
        ],
        experiment_prefix="offline-compliance-check"
    )
    
    print("\n✅ Offline evaluation complete. Check LangSmith dashboard for results.")

if __name__ == "__main__":
    run_offline_evaluation()
```

---

### Step 2: Custom Evaluators (`custom_evaluators.py`)

Reusable guardrail logic for deterministic checks.

```python
"""
Custom Evaluators for LangSmith

Includes:
- PII/Security guardrails (regex-based)
- JSON schema validation
- Token count monitoring
- Cost tracking

These are pure Python functions that integrate directly into LangSmith's evaluators array.
"""

import re
import json
import os

def security_gate(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Deterministic security check: fail if output contains SSN, API keys, or passwords.
    """
    response = outputs.get("answer", "")
    
    # Define patterns for sensitive data
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "api_key_openai": r"sk_(live|test)_[a-zA-Z0-9]{20,}",
        "password": r"(?i)password\s*=\s*['\"]?\w+['\"]?",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
    }
    
    found_issues = []
    for issue_type, pattern in patterns.items():
        if re.search(pattern, response):
            found_issues.append(issue_type)
    
    if found_issues:
        return {
            "key": "security_check",
            "score": 0.0,
            "comment": f"CRITICAL: Detected {', '.join(found_issues)} in output."
        }
    
    return {
        "key": "security_check",
        "score": 1.0,
        "comment": "✅ No sensitive data detected."
    }

def json_format_check(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Validate that output is valid JSON (if applicable).
    """
    response = outputs.get("answer", "")
    
    # Try to parse as JSON
    try:
        data = json.loads(response)
        return {
            "key": "json_format",
            "score": 1.0,
            "comment": "✅ Valid JSON structure."
        }
    except json.JSONDecodeError:
        # Not JSON (acceptable for text responses)
        if len(response) > 10 and response[0] != "{":
            return {
                "key": "json_format",
                "score": 1.0,
                "comment": "✅ Plain text response (JSON not required)."
            }
        
        return {
            "key": "json_format",
            "score": 0.0,
            "comment": "❌ Invalid JSON structure."
        }

def token_count_monitor(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Track token usage; flag if exceeds threshold.
    """
    # This would integrate with LangSmith's usage tracking
    # For demo, returning a placeholder
    return {
        "key": "token_efficiency",
        "score": 1.0,
        "comment": "✅ Token usage within budget."
    }

def tool_calling_validator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Validate tool invocation metadata: name, arguments, types.
    """
    generated_calls = outputs.get("tool_calls", [])
    expected = reference_outputs.get("reference_tool_call")
    
    if not expected:
        return {
            "key": "tool_validation",
            "score": 1.0,
            "comment": "⏭️ No tool expected for this input."
        }
    
    if not generated_calls:
        return {
            "key": "tool_validation",
            "score": 0.0,
            "comment": "❌ Expected tool call but none was made."
        }
    
    actual = generated_calls[0]
    issues = []
    
    # Check tool name
    if actual.get("name") != expected.get("name"):
        issues.append(f"Tool name: got '{actual.get('name')}', expected '{expected.get('name')}'")
    
    # Check argument keys
    expected_keys = set(expected.get("args", {}).keys())
    actual_keys = set(actual.get("args", {}).keys())
    missing_keys = expected_keys - actual_keys
    
    if missing_keys:
        issues.append(f"Missing arguments: {missing_keys}")
    
    if not issues:
        return {
            "key": "tool_validation",
            "score": 1.0,
            "comment": "✅ Tool invocation correct."
        }
    
    return {
        "key": "tool_validation",
        "score": 0.0,
        "comment": " | ".join(issues)
    }
```

---

### Step 3: Online Monitoring — Hallucination Detection (`test_online_monitoring.py`)

Production-grade self-consistency checks for detecting hallucinations.

```python
"""
Online Hallucination Detection via Self-Consistency

When a model is deployed to production, you lose access to human-labeled references.
This script implements reference-free hallucination detection by sampling multiple outputs
and checking for internal consistency.

If outputs diverge significantly, it signals potential hallucination.

Run: uv run test_online_monitoring.py
"""

import os
import truststore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np

truststore.inject_into_ssl()
load_dotenv()

def check_prerequisites():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set")
    print("✅ Prerequisites validated")

def hallucination_detector(user_question: str, primary_output: str = None):
    """
    Detect hallucination risk using self-consistency.
    
    Algorithm:
    1. If primary_output not provided, generate one (temperature=0)
    2. Sample 3–5 variations with high temperature
    3. Embed all outputs and compute pairwise cosine distances
    4. If variance > 0.25, flag as potential hallucination
    """
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    embedder = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Generate primary output if not provided
    if primary_output is None:
        print(f"🔹 Generating primary output for: '{user_question}'")
        response = model.invoke(user_question)
        primary_output = response.content
    
    print(f"📝 Primary output: {primary_output[:80]}...\n")
    
    # Sample high-temperature variations
    print("🔄 Sampling 3 high-temperature variations...")
    sampling_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.8,  # High variability
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    samples = []
    for i in range(3):
        response = sampling_model.invoke(user_question)
        samples.append(response.content)
        print(f"   Sample {i+1}: {response.content[:60]}...")
    
    # Embed and compute distances
    print("\n📊 Computing semantic variance...")
    primary_vec = embedder.embed_query(primary_output)
    sample_vecs = embedder.embed_documents(samples)
    
    distances = []
    for i, vec in enumerate(sample_vecs):
        # Cosine distance (1 - similarity)
        cos_sim = np.dot(primary_vec, vec) / (
            np.linalg.norm(primary_vec) * np.linalg.norm(vec)
        )
        cos_dist = 1.0 - cos_sim
        distances.append(cos_dist)
        print(f"   Distance to sample {i+1}: {cos_dist:.3f}")
    
    avg_variance = sum(distances) / len(distances)
    print(f"\n📈 Average variance: {avg_variance:.3f}")
    
    # Decision threshold
    HALLUCINATION_THRESHOLD = 0.25
    
    if avg_variance > HALLUCINATION_THRESHOLD:
        print(f"\n⚠️  HALLUCINATION ALERT: Variance {avg_variance:.3f} exceeds threshold {HALLUCINATION_THRESHOLD}")
        return {
            "key": "hallucination_risk",
            "score": 0.0,
            "variance": avg_variance,
            "comment": f"High internal divergence detected. Model may be hallucinating."
        }
    
    print(f"\n✅ STABLE OUTPUT: Variance {avg_variance:.3f} is within safe range")
    return {
        "key": "hallucination_risk",
        "score": 1.0,
        "variance": avg_variance,
        "comment": f"Output is internally consistent."
    }

if __name__ == "__main__":
    check_prerequisites()
    
    # Test hallucination detector
    test_questions = [
        "What is the capital of France?",
        "Explain quantum entanglement in one sentence.",
        "What did the CEO announce in last month's earnings call?"  # Likely to hallucinate if no context
    ]
    
    print("=" * 70)
    print("ONLINE HALLUCINATION DETECTION DEMO")
    print("=" * 70 + "\n")
    
    for question in test_questions:
        print(f"❓ Testing: {question}\n")
        result = hallucination_detector(question)
        print(f"Result: {result}\n")
        print("-" * 70 + "\n")
```

---

### Step 4: Tool Calling Evaluation (`test_tool_validation.py`)

Validate agent tool invocations deterministically.

```python
"""
Tool Calling Validation

Evaluates whether an agent invoked the correct tool with correct arguments.

Fault planes:
1. Selection Error — wrong tool name
2. Completeness Error — missing required arguments
3. Type Error — argument type mismatch

Run: uv run test_tool_validation.py
"""

import os
import truststore
from dotenv import load_dotenv

truststore.inject_into_ssl()
load_dotenv()

def validate_tool_invocation(actual_call: dict, expected_call: dict) -> dict:
    """
    Deterministic validation of tool invocation.
    
    Args:
        actual_call: {"name": "...", "args": {...}}
        expected_call: {"name": "...", "args": {...}, "arg_types": {...}}
    """
    issues = []
    
    # Fault plane 1: Selection error
    if actual_call.get("name") != expected_call.get("name"):
        issues.append({
            "type": "selection_error",
            "detail": f"Tool name: got '{actual_call.get('name')}', expected '{expected_call.get('name')}'"
        })
    
    # Fault plane 2: Completeness error
    expected_keys = set(expected_call.get("args", {}).keys())
    actual_keys = set(actual_call.get("args", {}).keys())
    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys
    
    if missing_keys:
        issues.append({
            "type": "completeness_error",
            "detail": f"Missing arguments: {missing_keys}"
        })
    
    if extra_keys:
        issues.append({
            "type": "extra_arguments",
            "detail": f"Unexpected arguments: {extra_keys}"
        })
    
    # Fault plane 3: Type errors
    expected_types = expected_call.get("arg_types", {})
    for arg_name, expected_type in expected_types.items():
        if arg_name in actual_call.get("args", {}):
            actual_value = actual_call["args"][arg_name]
            actual_type = type(actual_value).__name__
            if actual_type != expected_type:
                issues.append({
                    "type": "type_error",
                    "detail": f"Argument '{arg_name}': got {actual_type}, expected {expected_type}"
                })
    
    # Generate score
    if not issues:
        score = 1.0
        comment = "✅ Perfect tool invocation."
    else:
        # Deduct 0.33 per critical issue
        score = max(0.0, 1.0 - (len(issues) * 0.33))
        comment = " | ".join([issue["detail"] for issue in issues])
    
    return {
        "key": "tool_validation",
        "score": score,
        "issues": issues,
        "comment": comment
    }

if __name__ == "__main__":
    print("=" * 70)
    print("TOOL CALLING VALIDATION DEMO")
    print("=" * 70 + "\n")
    
    # Test case 1: Perfect invocation
    print("Test 1: Perfect tool call")
    result = validate_tool_invocation(
        actual_call={
            "name": "fetch_employee_policy",
            "args": {"policy_id": "int_remote_30", "include_amendments": True}
        },
        expected_call={
            "name": "fetch_employee_policy",
            "args": {"policy_id": "int_remote_30", "include_amendments": True},
            "arg_types": {"policy_id": "str", "include_amendments": "bool"}
        }
    )
    print(f"Result: {result}\n")
    
    # Test case 2: Selection error
    print("Test 2: Wrong tool selected")
    result = validate_tool_invocation(
        actual_call={
            "name": "fetch_wrong_policy",
            "args": {"policy_id": "int_remote_30"}
        },
        expected_call={
            "name": "fetch_employee_policy",
            "args": {"policy_id": "int_remote_30"},
            "arg_types": {"policy_id": "str"}
        }
    )
    print(f"Result: {result}\n")
    
    # Test case 3: Type error
    print("Test 3: Type mismatch")
    result = validate_tool_invocation(
        actual_call={
            "name": "fetch_employee_policy",
            "args": {"policy_id": 12345}  # Should be string
        },
        expected_call={
            "name": "fetch_employee_policy",
            "args": {"policy_id": "int_remote_30"},
            "arg_types": {"policy_id": "str"}
        }
    )
    print(f"Result: {result}\n")
    
    print("=" * 70)
```

---

## Beyond Basic

### Extended Examples

1. **[test_online_monitoring.py](src/test_online_monitoring.py)** — Self-consistency monitoring in production with multi-sample variance computation.

2. **[test_tool_validation.py](src/test_tool_validation.py)** — Comprehensive tool calling validator that catches selection, completeness, and type errors.

3. **[ingestion.py](src/ingestion.py)** — Bulk dataset initialization for 50+ golden examples.

### Recommended Next Steps

1. **Create a real golden dataset** — Add 10–20 of your actual production queries with human-verified answers.
2. **Integrate with CI/CD** — Run `uv run main.py` in your GitHub Actions / GitLab CI pipeline before deployment.
3. **Monitor costs** — Add cost tracking to your evaluators to prevent runaway inference bills.
4. **Threshold tuning** — Adjust hallucination variance threshold (currently 0.25) based on your production data.
5. **Tool catalog** — Expand `tool_metadata_check` to validate all tools your agent uses, not just one.

### Key Insights

- **Offline evaluation prevents regressions.** Always test before pushing to production.
- **Online evaluation catches drift.** Monitor self-consistency and hallucination risk in live traffic.
- **Deterministic guards scale.** Use regex/JSON validation for 100% of requests; sample semantic checks.
- **LangSmith is your dashboard.** Centralize traces, evaluations, and metrics in one place.
