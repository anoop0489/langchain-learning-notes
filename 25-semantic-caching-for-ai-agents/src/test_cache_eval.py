"""
Exact-Match vs Semantic Cache Evaluation

What this script does:
- Replays a labeled query stream.
- Compares exact-match caching against semantic caching.
- Reports hit rate, precision, recall, F1, and estimated savings.

Problem it solves:
- Teams often see semantic hit-rate gains but cannot quantify quality/savings impact.
- This script provides a compact before-vs-after baseline report.

Prerequisites:
- Python 3.12+
- Dependencies installed:
  uv add sentence-transformers python-dotenv truststore

Run:
- uv run 25-semantic-caching-for-ai-agents/src/test_cache_eval.py
"""

import os
import re
from collections import Counter
from dataclasses import dataclass

import truststore
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

truststore.inject_into_ssl()
load_dotenv()


def check_prerequisites() -> None:
    """Validate config values used in evaluation."""
    threshold = float(os.getenv("SEMCACHE_THRESHOLD", "0.86"))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("SEMCACHE_THRESHOLD must be in [0, 1]")


@dataclass
class EvalResult:
    hits: int = 0
    misses: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))


def fallback_embed(text: str, dim: int = 256) -> list[float]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    counts = Counter(tokens)
    vec = [0.0] * dim
    for token, count in counts.items():
        vec[hash(token) % dim] += float(count)

    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def compute_metrics(res: EvalResult) -> tuple[float, float, float, float]:
    total = res.hits + res.misses
    hit_rate = res.hits / total if total else 0.0
    precision = res.tp / (res.tp + res.fp) if (res.tp + res.fp) else 0.0
    recall = res.tp / (res.tp + res.fn) if (res.tp + res.fn) else 0.0
    f1 = 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return hit_rate, precision, recall, f1


def run_eval() -> None:
    model_name = os.getenv("SEMCACHE_EMBED_MODEL", "all-MiniLM-L6-v2")
    threshold = float(os.getenv("SEMCACHE_THRESHOLD", "0.86"))
    model = None
    using_fallback_embedder = False
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        using_fallback_embedder = True
        print(f"⚠️  Embedding model load failed ({exc.__class__.__name__}); using offline fallback embedder.")

    if using_fallback_embedder:
        threshold = float(os.getenv("SEMCACHE_FALLBACK_THRESHOLD", "0.45"))

    canonical_answers = {
        "refund": "Refunds are available within 30 days.",
        "shipping": "Shipping usually takes 3-5 business days.",
        "cancel": "Subscriptions can be canceled any time from settings.",
    }

    cache_seed = [
        ("How do I request a refund?", "refund"),
        ("Where is my order?", "shipping"),
        ("How do I cancel my subscription?", "cancel"),
    ]

    replay_stream = [
        ("I want my money back", "refund", 1),
        ("What is your refund policy?", "refund", 1),
        ("How to track delivery", "shipping", 1),
        ("Stop my recurring plan", "cancel", 1),
        ("What payment methods do you support?", "none", 0),
        ("Refund after 60 days", "none", 0),
        ("Where is my order?", "shipping", 1),
        ("How do I reset my password?", "none", 0),
    ]

    exact_map = {q.strip().lower(): label for q, label in cache_seed}

    semantic_entries: list[tuple[str, str, list[float]]] = []
    for q, label in cache_seed:
        if model is not None:
            vec = model.encode(q, normalize_embeddings=True).tolist()
        else:
            vec = fallback_embed(q)
        semantic_entries.append((q, label, vec))

    exact_res = EvalResult()
    sem_res = EvalResult()

    for query, expected_label, reusable in replay_stream:
        normalized = query.strip().lower()

        exact_hit = normalized in exact_map
        if exact_hit:
            exact_res.hits += 1
            predicted = exact_map[normalized]
            if reusable and predicted == expected_label:
                exact_res.tp += 1
            else:
                exact_res.fp += 1
        else:
            exact_res.misses += 1
            if reusable:
                exact_res.fn += 1

        if model is not None:
            q_vec = model.encode(query, normalize_embeddings=True).tolist()
        else:
            q_vec = fallback_embed(query)
        best_score = -1.0
        best_label = "none"

        for _, label, vec in semantic_entries:
            score = cosine_similarity(q_vec, vec)
            if score > best_score:
                best_score = score
                best_label = label

        sem_hit = best_score >= threshold
        if sem_hit:
            sem_res.hits += 1
            if reusable and best_label == expected_label:
                sem_res.tp += 1
            else:
                sem_res.fp += 1
        else:
            sem_res.misses += 1
            if reusable:
                sem_res.fn += 1

    exact_metrics = compute_metrics(exact_res)
    sem_metrics = compute_metrics(sem_res)

    baseline_latency_ms = float(os.getenv("SEMCACHE_BASELINE_LLM_LATENCY_MS", "900"))
    cache_latency_ms = float(os.getenv("SEMCACHE_CACHE_LATENCY_MS", "45"))
    llm_call_cost = float(os.getenv("SEMCACHE_COST_PER_LLM_CALL_USD", "0.003"))

    total_requests = len(replay_stream)
    exact_llm_calls = total_requests - exact_res.hits
    sem_llm_calls = total_requests - sem_res.hits

    exact_total_latency = exact_res.hits * cache_latency_ms + exact_llm_calls * baseline_latency_ms
    sem_total_latency = sem_res.hits * cache_latency_ms + sem_llm_calls * baseline_latency_ms

    exact_total_cost = exact_llm_calls * llm_call_cost
    sem_total_cost = sem_llm_calls * llm_call_cost

    print("=" * 92)
    print("Cache Evaluation: Exact Match vs Semantic")
    embedder_label = f"{model_name} (fallback)" if using_fallback_embedder else model_name
    print(f"Model={embedder_label} | threshold={threshold:.2f}")
    print("=" * 92)
    print("Approach      hit_rate  precision  recall    f1        llm_calls")
    print("-" * 92)
    print(
        f"Exact-match   {exact_metrics[0]:<8.3f}  {exact_metrics[1]:<9.3f}  "
        f"{exact_metrics[2]:<8.3f}  {exact_metrics[3]:<8.3f}  {exact_llm_calls}"
    )
    print(
        f"Semantic      {sem_metrics[0]:<8.3f}  {sem_metrics[1]:<9.3f}  "
        f"{sem_metrics[2]:<8.3f}  {sem_metrics[3]:<8.3f}  {sem_llm_calls}"
    )

    latency_saved_ms = exact_total_latency - sem_total_latency
    cost_saved = exact_total_cost - sem_total_cost

    print("\nEstimated Impact")
    print(f"Latency saved: {latency_saved_ms:.1f} ms over {total_requests} requests")
    print(f"Cost saved   : ${cost_saved:.4f} over {total_requests} requests")


if __name__ == "__main__":
    check_prerequisites()
    run_eval()
