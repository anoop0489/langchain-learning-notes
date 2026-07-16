# 40. Guardrails Implementation

## Project Structure

```text
27-guardrails/
├── 39_Guardrails_Theory_And_Concepts.md
├── 40_Guardrails_Implementation.md
├── assets/
└── src/
    ├── main.py
    ├── test_nli_vs_similarity.py
    ├── test_input_guardrails.py
    └── test_tool_guardrails.py
```

---

## Dependencies

Install required packages:

```bash
uv add langchain langchain-openai langsmith python-dotenv truststore
```

Optional packages for enterprise extensions:

```bash
uv add pydantic
```

Optional package if you want local embedding models for faster similarity experiments:

```bash
uv add sentence-transformers
```

---

## Environment Variables

Create or update `.env` in repo root:

```env
OPENAI_API_KEY=your_openai_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=guardrails-module
```

---

## Step-by-Step Walkthrough

### Step 1: Build a layered middleware stack

Goal:

1. Block obvious injection attempts before agent planning.
2. Redact sensitive data before model invocation.
3. Restrict risky tools with deterministic policy checks.
4. Apply final response filtering before release.

Key ideas:

- Start with deterministic filters (cheap, reliable).
- Add model-based checks only where rules are not enough.
- Put irreversible actions behind explicit approvals.

### Step 2: Implement inbound guardrails

Use middleware to inspect user messages before the model call:

- Keyword and phrase denylist.
- Tenant/role access validation.
- Input length and format checks.

### Step 3: Apply PII middleware

Use `PIIMiddleware` on input and output depending on policy:

- `redact` for compliance-safe prompts and logs.
- `mask` where user readability is useful.
- `block` for highly sensitive domains.

### Step 3A: Choose Scope Profile (Input-only vs Output-only vs Both)

Use one of these profiles per requirement:

#### Profile A: Input-only

Use when risk is mostly inbound (injection, off-domain requests, secret ingress).

```python
middleware=[
    InputPolicyMiddleware(),
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    PIIMiddleware("credit_card", strategy="block", apply_to_input=True),
    ToolPolicyMiddleware(),
]
```

#### Profile B: Output-only

Use when risk is primarily final response quality/safety/disclosure.

Note: `OutputSafetyMiddleware()` in the snippets below is a custom middleware you define for your project (for example, semantic safety checks in `after_agent`).

```python
middleware=[
    PIIMiddleware("email", strategy="redact", apply_to_output=True),
    PIIMiddleware("url", strategy="block", apply_to_output=True),
    OutputSafetyMiddleware(),
]
```

#### Profile C: Both input and output

Use for regulated or high-risk systems.

```python
middleware=[
    InputPolicyMiddleware(),
    PIIMiddleware("email", strategy="redact", apply_to_input=True),
    PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ToolPolicyMiddleware(),
    PIIMiddleware("email", strategy="redact", apply_to_output=True),
    OutputSafetyMiddleware(),
]
```

Profile selection guideline:

1. Low risk internal assistant: input-only may be enough.
2. Public chatbot with legal/compliance exposure: both is recommended.
3. Content moderation layer over trusted inputs: output-only can be acceptable.

### Step 4: Enforce tool policies

Before each tool call:

1. Verify tool is allowed for current user role.
2. Validate argument schema and safety bounds.
3. Deny side-effecting tools unless approved.

### Step 5: Trace and evaluate

Use LangSmith to track:

- Triggered guardrail type.
- Block reasons.
- Latency impact.
- False positive candidates.

### Step 6: Add NLI policy checks for nuanced requirements

Use NLI when deterministic logic cannot confidently classify the policy condition.

NLI pattern:

1. Define policy hypotheses (for example, "contains credential exfiltration request").
2. Compare observed text as premise against each hypothesis.
3. Use score thresholds to route block/review/allow.
4. Log label and confidence for auditability.

Example decision policy:

1. entailment >= 0.85: block
2. 0.60 <= entailment < 0.85: escalate for review
3. entailment < 0.60: allow unless another guardrail triggers

### Step 7: Add classifier cascade for scale and reliability

Use a tiered classifier architecture to balance cost, latency, and safety:

1. Tier A: fast broad screen on all traffic.
2. Tier B: stronger classifier only for uncertain/high-risk cases.
3. Tier C: HITL for high-impact ambiguous outcomes.

Implementation guidance:

1. Define confidence bands per policy label.
2. Map each band to an action (`allow`, `rewrite`, `review`, `block`).
3. Persist classifier model version and thresholds in trace metadata.
4. Recalibrate thresholds with offline datasets on a fixed cadence.

Example action mapping:

1. score >= 0.90: block
2. 0.70 <= score < 0.90: review
3. score < 0.70: allow with monitoring (low-risk flows)

### Step 8: Add the policy intelligence layer

Use a layered semantic decision stack when simple rules are not enough:

1. NER first to extract sensitive entities such as names, emails, orgs, IDs, and locations.
2. NLI next to evaluate policy hypotheses like "this text requests credential disclosure".
3. Semantic similarity for fuzzy matching, retrieval, and grouping similar policy incidents.
4. Classifier cascade for nuanced moderation labels such as toxicity, abuse, or policy violations.
5. HITL for the final unresolved high-impact cases.

Implementation guidance:

1. Treat NER as extraction, not enforcement.
2. Treat NLI as policy reasoning.
3. Treat similarity as a supporting signal, not a final block condition.
4. Treat classifiers as one stage in a cascade with calibrated thresholds.

Recommended workflow:

1. entity_detect -> 2. rule_check -> 3. nli_check -> 4. classifier_check -> 5. route_action

---

## Implementation: Core Secure Agent

The script in [src/main.py](src/main.py) provides:

1. Custom inbound filter middleware.
2. Built-in PII middleware stack.
3. Tool execution policy middleware.
4. Demonstration run with safe and unsafe prompts.

Run command:

```bash
uv run 27-guardrails/src/main.py
```

---

## Implementation: Input Guardrail Tests

The script in [src/test_input_guardrails.py](src/test_input_guardrails.py) tests:

1. Prompt injection patterns.
2. Off-domain requests.
3. PII redaction path.

Run command:

```bash
uv run 27-guardrails/src/test_input_guardrails.py
```

---

## Implementation: NLI vs Semantic Similarity Comparison

The script in [src/test_nli_vs_similarity.py](src/test_nli_vs_similarity.py) demonstrates:

1. Semantic similarity scoring with embeddings.
2. NLI-style entailment/contradiction classification using a policy hypothesis.
3. A case where similarity is high but NLI detects contradiction.

Run command:

```bash
uv run 27-guardrails/src/test_nli_vs_similarity.py
```

---

## Implementation: Tool Guardrail Tests

The script in [src/test_tool_guardrails.py](src/test_tool_guardrails.py) tests:

1. Tool allowlist behavior.
2. High-risk tool denial path.
3. Argument validation failure path.

Run command:

```bash
uv run 27-guardrails/src/test_tool_guardrails.py
```

---

## Beyond Basic

1. Add a policy version identifier to each guardrail decision and trace metadata.
2. Introduce shadow mode to measure false positives before enabling hard blocks.
3. Add a safety evaluator chain in `after_agent` for nuanced semantic checks.
4. Enforce region and data residency constraints in tool guardrails.
5. Build a correction loop that auto-generates test cases from blocked production traces.
6. Separate requirements into three policy files: `input_policies`, `output_policies`, and `shared_policies` to avoid accidental over/under enforcement.
7. Maintain a hypothesis registry for NLI guardrails and version it alongside policy releases.
8. Add a classifier registry with: model_id, label_set, thresholds, calibration_date, and owner.
9. Add a policy-intelligence registry with: entity types, NLI hypotheses, similarity use cases, and classifier labels.

---

## Production Checklist

1. Fail-closed for critical side effects.
2. PII checks applied to input, tool results, and output where required.
3. High-risk tools routed through HITL.
4. Full observability with tags and policy version metadata.
5. Rollout stages: shadow, soft, enforce.
6. Incident playbook and rollback path documented.
7. Every requirement is tagged with scope: `input-only`, `output-only`, or `both`.
8. Classifier cascade is implemented with explicit confidence-band actions.
9. Classifier thresholds are calibrated and versioned.
10. Policy-intelligence layer is explicitly assigned to extraction, reasoning, similarity, and moderation roles.
