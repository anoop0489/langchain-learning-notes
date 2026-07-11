"""
Semantic Cache Demo (End-to-End)

What this script does:
- Builds a minimal semantic cache for user queries.
- Uses exact-match first, then semantic similarity fallback.
- Simulates agent generation on cache miss and writes back to cache.

Problem it solves:
- Repeated user intent is often phrased differently.
- Exact string caches miss paraphrases and force extra model calls.
- Semantic matching improves reuse while preserving a safe fallback path.

Prerequisites:
- Python 3.12+
- Dependencies installed:
  uv add sentence-transformers python-dotenv truststore
- Optional .env values:
  SEMCACHE_EMBED_MODEL=all-MiniLM-L6-v2
  SEMCACHE_THRESHOLD=0.86

Run:
- uv run 25-semantic-caching-for-ai-agents/src/main.py
"""

import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import truststore
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

truststore.inject_into_ssl()
load_dotenv()


def check_prerequisites() -> None:
    """Validate runtime prerequisites and configuration sanity."""
    threshold_str = os.getenv("SEMCACHE_THRESHOLD", "0.86")
    try:
        threshold = float(threshold_str)
    except ValueError as exc:
        raise ValueError("SEMCACHE_THRESHOLD must be a valid float") from exc

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("SEMCACHE_THRESHOLD must be in [0.0, 1.0]")


@dataclass
class CacheEntry:
    query: str
    answer: str
    embedding: list[float]
    created_at: str


class SemanticCache:
    def __init__(self, model_name: str, threshold: float) -> None:
        self.model_name = model_name
        self.model = None
        self.using_fallback_embedder = False
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            # Corporate/restricted networks may block model downloads from HF.
            self.using_fallback_embedder = True
            print(f"⚠️  Embedding model load failed ({exc.__class__.__name__}); using offline fallback embedder.")
        self.threshold = threshold
        self.entries: list[CacheEntry] = []
        self.exact_map: dict[str, int] = {}
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0}

    @staticmethod
    def _fallback_embed(text: str, dim: int = 256) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        counts = Counter(tokens)
        vec = [0.0] * dim
        for token, count in counts.items():
            vec[hash(token) % dim] += float(count)

        norm = sum(x * x for x in vec) ** 0.5
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    def embed(self, text: str) -> list[float]:
        if self.model is not None:
            vector = self.model.encode(text, normalize_embeddings=True)
            return vector.tolist()
        return self._fallback_embed(text)

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        return float(sum(a * b for a, b in zip(vec_a, vec_b)))

    def store(self, query: str, answer: str) -> None:
        idx = len(self.entries)
        entry = CacheEntry(
            query=query,
            answer=answer,
            embedding=self.embed(query),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.entries.append(entry)
        self.exact_map[query.strip().lower()] = idx

    def check(self, query: str) -> tuple[str | None, str, float]:
        normalized = query.strip().lower()
        if normalized in self.exact_map:
            self.stats["exact_hits"] += 1
            idx = self.exact_map[normalized]
            return self.entries[idx].answer, "exact_hit", 1.0

        if not self.entries:
            self.stats["misses"] += 1
            return None, "miss", 0.0

        q_vec = self.embed(query)
        best_score = -1.0
        best_entry: CacheEntry | None = None

        for entry in self.entries:
            score = self.cosine_similarity(q_vec, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= self.threshold:
            self.stats["semantic_hits"] += 1
            return best_entry.answer, "semantic_hit", best_score

        self.stats["misses"] += 1
        return None, "miss", best_score


def fallback_agent_response(query: str) -> str:
    """Simulated fallback path for misses (replace with real RAG/agent call)."""
    knowledge = {
        "refund": "Refunds are available within 30 days with valid proof of purchase.",
        "shipping": "Standard shipping takes 3-5 business days.",
        "cancel": "Subscriptions can be canceled from account settings.",
    }

    lower = query.lower()
    for key, answer in knowledge.items():
        if key in lower:
            return answer

    return "I will route this to the full assistant pipeline for a fresh answer."


def run_demo() -> None:
    model_name = os.getenv("SEMCACHE_EMBED_MODEL", "all-MiniLM-L6-v2")
    threshold = float(os.getenv("SEMCACHE_THRESHOLD", "0.86"))

    cache = SemanticCache(model_name=model_name, threshold=threshold)
    if cache.using_fallback_embedder:
        fallback_threshold = float(os.getenv("SEMCACHE_FALLBACK_THRESHOLD", "0.45"))
        cache.threshold = fallback_threshold

    warmup_pairs = [
        ("How do I request a refund?", "Refunds are available within 30 days with valid proof of purchase."),
        ("How long does delivery take?", "Standard shipping takes 3-5 business days."),
        ("How do I cancel my subscription?", "Subscriptions can be canceled from account settings."),
    ]

    for q, a in warmup_pairs:
        cache.store(q, a)

    test_queries = [
        "I want my money back",
        "When will my order arrive?",
        "Stop my recurring plan",
        "How can I change my account email?",
        "How do I request a refund?",
    ]

    print("=" * 72)
    print("Semantic Cache Demo")
    print(f"Model: {model_name}")
    print(f"Threshold: {cache.threshold}")
    if cache.using_fallback_embedder:
        print("Embedder: offline fallback (model download unavailable)")
    print("=" * 72)

    for query in test_queries:
        cached_answer, route, score = cache.check(query)
        if cached_answer is not None:
            print(f"\n✅ {route.upper()} | score={score:.3f}")
            print(f"Q: {query}")
            print(f"A: {cached_answer}")
        else:
            answer = fallback_agent_response(query)
            cache.store(query, answer)
            print(f"\n🟡 MISS | best_score={score:.3f}")
            print(f"Q: {query}")
            print(f"A: {answer}")

    print("\n" + "-" * 72)
    print("Run Stats")
    print(f"Exact hits   : {cache.stats['exact_hits']}")
    print(f"Semantic hits: {cache.stats['semantic_hits']}")
    print(f"Misses       : {cache.stats['misses']}")
    print(f"Entries      : {len(cache.entries)}")


if __name__ == "__main__":
    check_prerequisites()
    run_demo()
