Here is a comprehensive, production-grade `EVALUATION_GUIDE.md` file. It compiles every single concept, code architecture blueprint, and strategic pattern we have discussed into a single, cohesive, authoritative reference manual.

You can save this directly into your team's engineering repository as an official internal technical document.

---

# Technical Guide: Enterprise LLM & Agent Evaluation Architecture

## 1. The Paradigm Shift: Deterministic vs. Non-Deterministic Testing

Traditional software engineering relies on **deterministic assertions**: given a specific input, the system must output an exact, predictable value ($2 + 2 = 4$).

LLMs and Agentic AI applications operate **non-deterministically**. The same query executed across identical code models can produce distinct text paths that are all factually correct and semantically valid. Therefore, validation suites must shift from rigid text-string assertions to **semantic scoring matrices** and **structured metadata validation**.

---

## 2. Global Architectural Framework

An enterprise evaluation pipeline is bifurcated into two independent tracking lifecycles:

* **Offline Evaluation (Pre-Deployment CI/CD):** Testing static target code configurations against a fixed, human-verified validation protocol (**Golden Dataset**) before shipping features to production.
* **Online Evaluation (Production Telemetry):** Intercepting live user payloads, tracing run logs, and applying reference-free checks to ensure runtime performance doesn't suffer from data drift or hallucinations.

---

## 3. Offline Evaluation Implementation Blueprint

Offline execution requires a baseline dataset populated with two foundational pillars:

1. **Input:** The raw prompt or payload passed to the application.
2. **Reference (Ground Truth):** The ideal, human-verified correct answer or target schema.

### Core Architecture: LLM-as-a-Judge vs. Code-Based Vector Distances

* **LLM-as-a-Judge:** Utilizes an isolated, high-tier model (e.g., `gemini-1.5-pro`) running specialized testing prompts (e.g., `correctness`) to grade the live output against the ground truth using natural language reasoning.
* **Functional Code/Vector Distances:** Bypasses LLM judgment entirely. Converts strings into mathematical arrays via an embedding model and computes a geometric value (e.g., Cosine Distance) to score alignment.

### Production Pipeline Code

```python
import os
import numpy as np
from langsmith import Client, evaluate
from langchain_google_genai import ChatGoogleGenAI, GoogleGenAIEmbeddings

# Initialize baseline clients
client = Client()
DATASET_NAME = "Enterprise_Compliance_Golden_v1"

# =====================================================================
# STEP 1: Programmatic Dataset Initialization
# =====================================================================
if not client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Production reference dataset for core compliance gates."
    )
    client.create_examples(
        inputs=[
            {"question": "What is the policy on international remote work?"},
            {"question": "What happens if an expense report is submitted late?"}
        ],
        outputs=[
            {"reference": "Employees may work internationally up to 30 calendar days per year with prior manager approval."},
            {"reference": "Late expense submissions are deferred to the next bi-weekly payroll processing cycle."}
        ],
        dataset_id=dataset.id
    )

# =====================================================================
# STEP 2: Core Target Application (Simulated Production Bot)
# =====================================================================
def compliance_agent_target(inputs: dict) -> dict:
    user_query = inputs["question"]
    # App deployment model
    model = ChatGoogleGenAI(model="gemini-3.5-flash", temperature=0.0)
    response = model.invoke(f"Answer using formal corporate compliance rules: {user_query}")
    return {"output": response.content}

# =====================================================================
# STEP 3: Execution of Mixed Evaluators
# =====================================================================
def run_offline_test_suite():
    print("🚀 Triggering Offline Testing Regression Matrix...")
    
    # Run the evaluation harness pulling down the dataset
    evaluate(
        compliance_agent_target,
        data=DATASET_NAME,
        evaluators=[
            "correctness"  # Pre-baked LLM-as-a-Judge validation checking factual intent
        ],
        evaluator_config={
            "evaluator_llm": ChatGoogleGenAI(model="gemini-1.5-pro", temperature=0.0)
        },
        experiment_prefix="ci-cd-compliance-run"
    )
    print("🎉 Evaluation pipeline run synced successfully to dashboard logs.")

if __name__ == "__main__":
    run_offline_test_suite()

```

---

## 4. Custom Code Evaluators (Deterministic Logic Gates)

You can pass entirely custom Python functions to intercept evaluation payloads. This allows you to enforce strict business rules without relying on an LLM to think.

### Implementation Examples

```python
import re
import json

# 1. PII & Security Guardrail Evaluator
def security_and_pii_gate(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    bot_response = outputs.get("output", "")
    
    # Define corporate regex patterns
    ssn_regex = r"\b\d{3}-\d{2}-\d{4}\b"
    password_regex = r"(?i)password\s*=\s*['\"]?\w+['\"]?"
    
    if re.search(ssn_regex, bot_response) or re.search(password_regex, bot_response):
        return {
            "key": "security_safety_gate",
            "score": 0.0,
            "comment": "CRITICAL RISK: PII data leakage or credentials detected in output stream."
        }
    return {"key": "security_safety_gate", "score": 1.0, "comment": "Secure payload syntax verified."}

# 2. Structured JSON Format Guardrail
def json_validity_gate(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    bot_response = outputs.get("output", "")
    try:
        json.loads(bot_response)
        return {"key": "json_validation", "score": 1.0, "comment": "Valid structural JSON format."}
    except json.JSONDecodeError:
        return {"key": "json_validation", "score": 0.0, "comment": "Output formatting broken; invalid JSON payload."}

```

To run these, simply append the function identifiers directly into the `evaluators` execution property array: `evaluators=["correctness", security_and_pii_gate, json_validity_gate]`.

---

## 5. Online Evaluation Architecture (Live Traffic Monitoring)

When code models go live, you lose access to human-verified target references. Online metrics must change to **Reference-Free Evaluations**.

### The Self-Consistency Pattern (Anti-Hallucination Guardrail)

To catch deep factual hallucinations without a ground-truth document, systems track **mathematical response stability**.

If a model accurately knows a fact, it will consistently reproduce the same core meaning across high-randomness variants. If it is guessing or hallucinating, its outputs will drift wildly.

```python
def self_consistency_monitor(inputs: dict, outputs: dict) -> dict:
    user_prompt = inputs.get("question")
    actual_production_output = outputs.get("output")
    
    # Setup testing models
    sampling_engine = ChatGoogleGenAI(model="gemini-3.5-flash", temperature=0.8)
    embedder = GoogleGenAIEmbeddings(model="models/text-embedding-004")
    
    # Sample background iterations to test internal model agreement
    samples = [sampling_engine.invoke(user_prompt).content for _ in range(3)]
    
    # Compute vector alignment metrics
    base_vec = embedder.embed_query(actual_production_output)
    sample_vecs = embedder.embed_documents(samples)
    
    distances = []
    for vec in sample_vecs:
        cos_dist = 1.0 - np.dot(base_vec, vec) / (np.linalg.norm(base_vec) * np.linalg.norm(vec))
        distances.append(cos_dist)
        
    avg_variance = sum(distances) / len(distances)
    
    # An average cosine distance variance > 0.25 implies structural semantic divergence
    if avg_variance > 0.25:
        return {"key": "hallucination_index", "score": 0.0, "comment": f"High probability hallucination drift: {avg_variance:.2f}"}
    return {"key": "hallucination_index", "score": 1.0, "comment": f"Stable consensus verified: {avg_variance:.2f}"}

```

---

## 6. Tool-Calling Evaluation (Structured Schema Gates)

Evaluating function or tool calling requires checking **metadata schemas** rather than tracking verbal context or sentiments.

### The Three Operational Fault Planes

1. **Selection Errors:** Evaluating whether the agent targeted the correct operation name string.
2. **Completeness Errors:** Verifying that all mandatory dictionary keys specified by your codebase signature were successfully provided.
3. **Type Signature Errors:** Ensuring variable arguments conform exactly to typing parameters (e.g., verifying an account identifier is generated as a standard `int` instead of a textual string).

### Functional Metadata Evaluator Code

```python
def validate_tool_metadata(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Performs deterministic unit validations across functional tool invocations."""
    generated_calls = outputs.get("tool_calls", [])
    expected_schema = reference_outputs.get("reference_tool_call") # Stored JSON layout
    
    if not generated_calls:
        return {"key": "tool_gate", "score": 0.0, "comment": "System failed to call any tools."}
        
    actual_call = generated_calls[0]
    
    # Check 1: Exact Name Alignment
    if actual_call["name"] != expected_schema["name"]:
        return {"key": "tool_gate", "score": 0.0, "comment": f"Wrong tool selected. Target: {expected_schema['name']}."}
        
    # Check 2: Strict Dictionary Argument Parameter Verification
    if actual_call["args"] == expected_schema["args"]:
        return {"key": "tool_gate", "score": 1.0, "comment": "Complete validation match across parameters and values."}
        
    # Check 3: Soft Fallback Parameter Structural Integrity Check
    if set(actual_call["args"].keys()) == set(expected_schema["args"].keys()):
        return {"key": "tool_gate", "score": 0.5, "comment": "Keys match, but runtime argument data holds structural variance."}
        
    return {"key": "tool_gate", "score": 0.0, "comment": "Total argument signature compilation mismatch."}

```

---

## 7. Operational Summary & Best Practices Matrix

| Evaluation Layer | Core Focus | Primary Check Technique | Execution Triggers |
| --- | --- | --- | --- |
| **Offline Semantic** | Factual correctness, policy rules | LLM-as-a-Judge (`correctness`) / Embedding Vector Distance | Continuous Integration (CI/CD) pipelines |
| **Offline Structural** | Validating internal tool routing structures | Exact Python dictionary match against target schema | Code changes, database prompt alterations |
| **Online Guardrails** | System safety, protecting internal data assets | Deterministic Custom Code Evaluators (Regex scanners) | 100% of live application traffic traces |
| **Online Semantic** | Hallucination avoidance, handling tone drift | Self-Consistency checks / RAG Groundedness prompts | Sampled traffic chunks (e.g., 10% of runtime data logs) |