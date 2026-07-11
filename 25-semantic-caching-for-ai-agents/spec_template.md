# Semantic Cache System Spec Template

Use this template to define your semantic caching strategy before implementation. Keep it concise and explicit.

## 1) System Identity

- System Name:
- Domain / Use Case:
- Primary Users:
- Risk Level (low / medium / high):

## 2) Purpose and Scope

- Purpose (1-2 sentences):
- In-scope query categories:
- Out-of-scope query categories:
- Non-cacheable intents (if any):

## 3) Embedding Strategy

- Embedding model:
- Embedding dimension:
- Similarity metric (cosine / dot / l2):
- Rationale for model choice:
- Known limitations of this model in your domain:

## 4) Cache Entry Design

- Cached value type (final answer / structured response / intermediate output):
- Required metadata fields:
  - tenant_id:
  - locale:
  - policy_version:
  - role / entitlement:
  - knowledge_snapshot:
- TTL policy:
- Invalidation triggers:

## 5) Hit Decision Policy

- Exact-match first? (yes/no):
- Semantic threshold:
- Additional acceptance checks:
  - tenant boundary check:
  - locale check:
  - policy version check:
  - optional verifier:
- High-risk handling rule:

## 6) Miss Handling

- Fallback pipeline (RAG / agent / tool chain):
- Cache write trigger (when to store response):
- What must never be cached:

## 7) Evaluation Plan

- Offline labeled dataset source:
- Evaluation metrics:
  - hit rate:
  - precision:
  - recall:
  - F1:
  - p50/p95 latency:
  - estimated LLM call reduction:
- Precision floor target:
- Threshold sweep range:

## 8) Observability and Ops

- Required logs per decision:
- Dashboards:
- Alert thresholds:
- Rollout plan (shadow mode / canary / phased):
- Embedding model migration plan:

## 9) Security and Privacy

- Tenant isolation strategy:
- PII handling:
- Data retention policy:
- Access-control assumptions:

## 10) Example Query Pairs

### Should hit (answer-equivalent)
- 
- 
- 

### Should miss (not answer-equivalent)
- 
- 
- 

## 11) Sign-off Checklist

- [ ] Precision floor validated offline
- [ ] False-positive examples reviewed manually
- [ ] Freshness + invalidation policy tested
- [ ] Tenant/authorization boundaries tested
- [ ] Shadow-mode results reviewed
- [ ] Production rollout approved
