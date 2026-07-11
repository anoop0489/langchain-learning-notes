# 37. Semantic Caching Implementation — From Design to Evaluation

Step-by-step implementation guide for building, tuning, and evaluating a production-style semantic cache for AI agents.

*Based on Section 25 theory note: `36_Semantic_Caching_For_AI_Agents.md`*

---

## Project Overview

In this section, we implement a practical semantic cache workflow that you can adapt to support agents, RAG assistants, and FAQ-heavy systems.

The implementation objective is simple:

1. Reuse prior answers for semantically equivalent queries.
2. Minimize incorrect cache hits with guard rails and threshold tuning.
3. Measure business value using hit rate, precision, recall, F1, latency, and estimated cost savings.

### Project Structure

```text
25-semantic-caching-for-ai-agents/
├── 36_Semantic_Caching_For_AI_Agents.md      # Theory and production concepts
├── 37_Semantic_Caching_Implementation.md      # This file (implementation walkthrough)
├── spec_template.md                            # Domain/spec template for cache design
└── src/
    ├── main.py                                 # End-to-end semantic cache demo
    ├── main_redis.py                           # RedisVL-backed semantic cache demo
    ├── test_threshold_sweep.py                 # Offline threshold tuning (precision/recall/F1)
    └── test_cache_eval.py                      # Exact-match vs semantic-cache evaluation
```

---

## Dependencies

```bash
uv add sentence-transformers python-dotenv truststore redis redisvl
```

Why this split:

- `main.py` and local evaluation scripts run with local embeddings (`sentence-transformers`) and do not require Redis.
- `main_redis.py` demonstrates a RedisVL-backed path for shared production cache infrastructure.

---

## Environment Variables (`.env`)

```bash
# Embedding model for local semantic matching
SEMCACHE_EMBED_MODEL=all-MiniLM-L6-v2

# Hit decision threshold (cosine similarity)
SEMCACHE_THRESHOLD=0.86
SEMCACHE_FALLBACK_THRESHOLD=0.45

# Optional: latency/cost assumptions for quick ROI estimate
SEMCACHE_BASELINE_LLM_LATENCY_MS=900
SEMCACHE_CACHE_LATENCY_MS=45
SEMCACHE_COST_PER_LLM_CALL_USD=0.003

# Redis-backed semantic cache path
REDIS_URL=redis://localhost:6379
REDIS_CACHE_NAME=semantic_cache_demo
REDIS_DISTANCE_THRESHOLD=0.13
REDIS_CACHE_TTL=604800
```

`SEMCACHE_FALLBACK_THRESHOLD` is used only when the embedding model cannot be downloaded (for example, restricted corporate network). It keeps local demos runnable while reducing false-positive hits from the lightweight fallback embedder.

---

## Implementation Phases

## Phase 1: Define the cache contract

Before code, lock down your cache contract in a spec:

- What responses are safe to reuse?
- What metadata boundaries must match (tenant, locale, policy version)?
- What TTL/freshness policy applies?
- What is your precision floor?

Use `spec_template.md` and tailor it to your domain.

---

## Phase 2: Build baseline semantic cache flow (`src/main.py`)

`main.py` demonstrates:

1. exact-match fast path
2. semantic nearest-neighbor match
3. similarity threshold decision
4. cache write on miss
5. simulated agent fallback

Core flow:

```text
query -> exact lookup -> semantic lookup -> threshold check -> hit or fallback -> store
```

Key design details:

- Embeddings are normalized to simplify cosine similarity scoring.
- A small in-memory index keeps the example dependency-light.
- The script tracks cache metrics so you can observe behavior immediately.

Run:

```bash
uv run 25-semantic-caching-for-ai-agents/src/main.py
```

---

## Phase 3: Tune threshold offline (`src/test_threshold_sweep.py`)

A semantic cache without threshold tuning is unsafe.

This script:

1. uses a labeled equivalence set
2. sweeps thresholds across a range
3. computes precision, recall, and F1 at each threshold
4. prints the top thresholds by F1 and by precision

Run:

```bash
uv run 25-semantic-caching-for-ai-agents/src/test_threshold_sweep.py
```

How to use output:

- High-risk use cases: pick a threshold that satisfies a precision floor.
- Low-risk FAQ use cases: pick a threshold with better recall while retaining acceptable precision.

---

## Phase 4: Redis-backed semantic cache (`src/main_redis.py`)

`main_redis.py` demonstrates:

1. Redis connection and health check.
2. Semantic cache read/write using RedisVL.
3. Cache-hit short-circuit behavior.
4. Miss fallback + write-back pattern.

Run:

```bash
uv run 25-semantic-caching-for-ai-agents/src/main_redis.py
```

If Redis is not running locally, start Redis Stack:

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

---

## Phase 5: Evaluate business impact (`src/test_cache_eval.py`)

This script compares:

- exact-match cache baseline
- semantic cache

It reports:

- cache hit rate
- precision/recall/F1 from a labeled replay
- estimated latency savings
- estimated LLM-call reduction

Run:

```bash
uv run 25-semantic-caching-for-ai-agents/src/test_cache_eval.py
```

Use this as your pre-production checklist:

1. no unacceptable precision regression
2. measurable hit-rate gain over exact cache
3. meaningful latency and cost improvement

---

## Production Upgrade Path

After local validation, productionize in layers:

1. Replace in-memory entries with Redis/RedisVL vector index.
2. Add tenant/locale/policy filters in cache candidate search.
3. Add TTL + policy version invalidation.
4. Add a second-stage verifier (re-ranker or compact LLM check).
5. Add full tracing/monitoring with your observability stack.

---

## Example Decision Policy (Practical)

```text
Accept semantic hit only if:
- similarity >= threshold
- tenant_id matches
- locale matches
- policy_version matches
- request is not high-risk
Otherwise: fallback to fresh generation.
```

This policy prevents the most common production mistakes:

- cross-tenant leakage
- stale policy answers
- over-aggressive reuse on risky intents

---

## Beyond Basic

After this implementation, extend with:

1. Multi-tenant partitioning and policy-version namespaces.
2. Re-ranker stage for stronger false-positive control.
3. Shadow-mode rollout before enabling live semantic hits.
4. Agent subtask-level semantic caching (not just top-level query caching).

Related files:

- `src/main.py`
- `src/main_redis.py`
- `src/test_threshold_sweep.py`
- `src/test_cache_eval.py`
- `spec_template.md`
