"""
Redis Semantic Cache Demo (Production Path)

What this script does:
- Demonstrates semantic cache checks/writes using RedisVL SemanticCache.
- Falls back to a simulated agent response on cache miss.
- Prints cache route decisions and basic run stats.

Problem it solves:
- Moves Section 25 from in-memory demonstration to a Redis-backed cache path.
- Helps validate production-like cache behavior with shared infrastructure.

Prerequisites:
- Python 3.12+
- Dependencies installed:
  uv add redis redisvl python-dotenv truststore
- Redis running (example):
  docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
- Optional .env values:
  REDIS_URL=redis://localhost:6379
  REDIS_CACHE_NAME=semantic_cache_demo
  REDIS_DISTANCE_THRESHOLD=0.13
  REDIS_CACHE_TTL=604800

Run:
- uv run 25-semantic-caching-for-ai-agents/src/main_redis.py
"""

import os
from typing import Any

import truststore
from dotenv import load_dotenv

truststore.inject_into_ssl()
load_dotenv()


def check_prerequisites() -> None:
    """Validate required environment values and numeric ranges."""
    redis_url = os.getenv("REDIS_URL", "").strip()
    threshold_str = os.getenv("REDIS_DISTANCE_THRESHOLD", "0.13")
    ttl_str = os.getenv("REDIS_CACHE_TTL", "604800")

    if not redis_url:
        raise ValueError(
            "REDIS_URL is required. Example: rediss://<host>:6379 (or include auth if required)."
        )

    try:
        threshold = float(threshold_str)
    except ValueError as exc:
        raise ValueError("REDIS_DISTANCE_THRESHOLD must be a valid float") from exc

    if threshold < 0.0:
        raise ValueError("REDIS_DISTANCE_THRESHOLD must be >= 0")

    try:
        ttl = int(ttl_str)
    except ValueError as exc:
        raise ValueError("REDIS_CACHE_TTL must be a valid integer") from exc

    if ttl <= 0:
        raise ValueError("REDIS_CACHE_TTL must be > 0")

    connect_timeout_str = os.getenv("REDIS_CONNECT_TIMEOUT_SEC", "5")
    read_timeout_str = os.getenv("REDIS_READ_TIMEOUT_SEC", "5")
    max_retries_str = os.getenv("REDIS_MAX_RETRIES", "1")

    try:
        connect_timeout = float(connect_timeout_str)
    except ValueError as exc:
        raise ValueError("REDIS_CONNECT_TIMEOUT_SEC must be a valid float") from exc
    if connect_timeout <= 0:
        raise ValueError("REDIS_CONNECT_TIMEOUT_SEC must be > 0")

    try:
        read_timeout = float(read_timeout_str)
    except ValueError as exc:
        raise ValueError("REDIS_READ_TIMEOUT_SEC must be a valid float") from exc
    if read_timeout <= 0:
        raise ValueError("REDIS_READ_TIMEOUT_SEC must be > 0")

    try:
        max_retries = int(max_retries_str)
    except ValueError as exc:
        raise ValueError("REDIS_MAX_RETRIES must be a valid integer") from exc
    if max_retries < 0:
        raise ValueError("REDIS_MAX_RETRIES must be >= 0")


def fallback_agent_response(query: str) -> str:
    """Simulated miss path (replace with real RAG/agent call in production)."""
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


def extract_response_from_match(match: Any) -> str | None:
    """Safely extract response text from varying RedisVL match shapes."""
    if match is None:
        return None

    if isinstance(match, dict):
        return (
            match.get("response")
            or match.get("answer")
            or match.get("metadata", {}).get("response")
        )

    response = getattr(match, "response", None)
    if isinstance(response, str):
        return response

    metadata = getattr(match, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get("response") or metadata.get("answer")

    return None


def get_top_match(raw_result: Any) -> Any:
    """Normalize RedisVL check() output into a top-match object/dict."""
    if raw_result is None:
        return None

    if isinstance(raw_result, list):
        return raw_result[0] if raw_result else None

    matches = getattr(raw_result, "matches", None)
    if isinstance(matches, list):
        return matches[0] if matches else None

    if isinstance(raw_result, dict):
        matches_dict = raw_result.get("matches")
        if isinstance(matches_dict, list) and matches_dict:
            return matches_dict[0]

    return raw_result


def run_demo() -> None:
    try:
        import redis
        from redisvl.extensions.cache.llm import SemanticCache
    except Exception as exc:
        raise RuntimeError(
            "Missing Redis dependencies. Install with: uv add redis redisvl"
        ) from exc

    redis_url = os.getenv("REDIS_URL", "").strip()
    cache_name = os.getenv("REDIS_CACHE_NAME", "semantic_cache_demo")
    distance_threshold = float(os.getenv("REDIS_DISTANCE_THRESHOLD", "0.13"))
    ttl = int(os.getenv("REDIS_CACHE_TTL", "604800"))
    connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT_SEC", "5"))
    read_timeout = float(os.getenv("REDIS_READ_TIMEOUT_SEC", "5"))
    max_retries = int(os.getenv("REDIS_MAX_RETRIES", "1"))

    try:
        print("⏳ Connecting to Redis...")
        client = redis.from_url(
            redis_url,
            socket_connect_timeout=connect_timeout,
            socket_timeout=read_timeout,
            retry_on_timeout=False,
            max_connections=4,
        )
        client.ping()
        print("✅ Redis ping succeeded.")
    except Exception as exc:
        print("❌ Redis connection failed.")
        print(f"REDIS_URL: {redis_url}")
        print(f"Error    : {exc}")
        print("Troubleshooting:")
        print("1) Confirm host/port and security-group/network rules.")
        print("2) If using AWS/managed Redis with TLS, use rediss://... (TLS).")
        print("3) If auth is enabled, include credentials in REDIS_URL.")
        print("4) For local testing only, start Redis Stack:")
        print("   docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest")
        return

    print("⏳ Initializing RedisVL SemanticCache...")
    cache = SemanticCache(
        name=cache_name,
        redis_url=redis_url,
        distance_threshold=distance_threshold,
        ttl=ttl,
        overwrite=False,
    )

    stats = {"hits": 0, "misses": 0}

    warmup_pairs = [
        ("How do I request a refund?", "Refunds are available within 30 days with valid proof of purchase."),
        ("How long does delivery take?", "Standard shipping takes 3-5 business days."),
        ("How do I cancel my subscription?", "Subscriptions can be canceled from account settings."),
    ]

    for q, a in warmup_pairs:
        cache.store(prompt=q, response=a)

    test_queries = [
        "I want my money back",
        "When will my order arrive?",
        "Stop my recurring plan",
        "How can I change my account email?",
        "How do I request a refund?",
    ]

    print("=" * 80)
    print("Redis Semantic Cache Demo")
    print(f"Redis URL        : {redis_url}")
    print(f"Cache Name       : {cache_name}")
    print(f"Distance Threshold: {distance_threshold}")
    print(f"TTL (seconds)    : {ttl}")
    print(f"Connect timeout  : {connect_timeout}s")
    print(f"Read timeout     : {read_timeout}s")
    print(f"Max retries      : {max_retries}")
    print("=" * 80)

    for query in test_queries:
        raw = cache.check(prompt=query)
        top_match = get_top_match(raw)
        cached_answer = extract_response_from_match(top_match)

        if isinstance(cached_answer, str) and cached_answer.strip():
            stats["hits"] += 1
            print("\n✅ REDIS_SEMANTIC_HIT")
            print(f"Q: {query}")
            print(f"A: {cached_answer}")
        else:
            stats["misses"] += 1
            answer = fallback_agent_response(query)
            cache.store(prompt=query, response=answer)
            print("\n🟡 REDIS_CACHE_MISS")
            print(f"Q: {query}")
            print(f"A: {answer}")

    print("\n" + "-" * 80)
    print("Run Stats")
    print(f"Hits   : {stats['hits']}")
    print(f"Misses : {stats['misses']}")


if __name__ == "__main__":
    check_prerequisites()
    run_demo()
