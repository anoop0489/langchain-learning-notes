"""
Threshold Sweep Evaluation for Semantic Cache

What this script does:
- Evaluates multiple similarity thresholds against labeled query pairs.
- Computes precision, recall, and F1 for each threshold.
- Helps choose a threshold based on safety (precision) vs savings (recall).

Problem it solves:
- Semantic caches fail when thresholds are chosen arbitrarily.
- This script makes threshold selection data-driven and reproducible.

Prerequisites:
- Python 3.12+
- Dependencies installed:
  uv add sentence-transformers python-dotenv truststore

Run:
- uv run 25-semantic-caching-for-ai-agents/src/test_threshold_sweep.py
"""

import os
import re
from collections import Counter

import truststore
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

truststore.inject_into_ssl()
load_dotenv()


def check_prerequisites() -> None:
    """Validate threshold sweep config ranges."""
    sweep_min = float(os.getenv("SWEEP_MIN", "0.70"))
    sweep_max = float(os.getenv("SWEEP_MAX", "0.95"))
    sweep_step = float(os.getenv("SWEEP_STEP", "0.01"))

    if not (0.0 <= sweep_min < sweep_max <= 1.0):
        raise ValueError("SWEEP_MIN and SWEEP_MAX must satisfy 0 <= min < max <= 1")
    if sweep_step <= 0:
        raise ValueError("SWEEP_STEP must be > 0")


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


def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def run_threshold_sweep() -> None:
    model_name = os.getenv("SEMCACHE_EMBED_MODEL", "all-MiniLM-L6-v2")
    model = None
    using_fallback_embedder = False
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        using_fallback_embedder = True
        print(f"⚠️  Embedding model load failed ({exc.__class__.__name__}); using offline fallback embedder.")

    labeled_pairs = [
        ("How do I request a refund?", "I want my money back", 1),
        ("Where is my order?", "How do I track my shipment?", 1),
        ("Cancel my subscription", "Stop recurring billing", 1),
        ("How do I request a refund?", "Do you offer gift cards?", 0),
        ("Where is my order?", "How to change payment method?", 0),
        ("Cancel my subscription", "How to reset my password?", 0),
        ("Refund in 30 days", "Refund in 60 days", 0),
        ("Python list sort", "Sort array in Python", 1),
        ("Python list sort", "Sort array in JavaScript", 0),
        ("How to delete account", "Can I permanently remove my profile?", 1),
    ]

    embedded = []
    for q1, q2, label in labeled_pairs:
        if model is not None:
            v1 = model.encode(q1, normalize_embeddings=True).tolist()
            v2 = model.encode(q2, normalize_embeddings=True).tolist()
        else:
            v1 = fallback_embed(q1)
            v2 = fallback_embed(q2)
        score = cosine_similarity(v1, v2)
        embedded.append((score, label))

    default_sweep_min = "0.15" if using_fallback_embedder else "0.70"
    default_sweep_max = "0.60" if using_fallback_embedder else "0.95"
    sweep_min = float(os.getenv("SWEEP_MIN", default_sweep_min))
    sweep_max = float(os.getenv("SWEEP_MAX", default_sweep_max))
    sweep_step = float(os.getenv("SWEEP_STEP", "0.01"))

    thresholds = []
    current = sweep_min
    while current <= sweep_max + 1e-12:
        thresholds.append(round(current, 4))
        current += sweep_step

    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for score, label in embedded:
            predicted_hit = score >= threshold
            if predicted_hit and label == 1:
                tp += 1
            elif predicted_hit and label == 0:
                fp += 1
            elif not predicted_hit and label == 1:
                fn += 1

        precision, recall, f1 = compute_metrics(tp, fp, fn)
        rows.append((threshold, precision, recall, f1, tp, fp, fn))

    best_f1 = max(rows, key=lambda r: r[3])
    best_precision = max(rows, key=lambda r: (r[1], r[2]))

    print("=" * 88)
    print("Threshold Sweep Results")
    embedder_label = f"{model_name} (fallback)" if using_fallback_embedder else model_name
    print(f"Model: {embedder_label}")
    print(f"Sweep range: [{sweep_min:.2f}, {sweep_max:.2f}] step={sweep_step:.2f}")
    print("=" * 88)
    print("thr    precision  recall    f1        tp  fp  fn")
    print("-" * 88)

    for threshold, precision, recall, f1, tp, fp, fn in rows:
        print(f"{threshold:<6.2f} {precision:<10.3f} {recall:<9.3f} {f1:<9.3f} {tp:<3d} {fp:<3d} {fn:<3d}")

    print("\nTop Picks")
    print(
        f"Best F1       -> thr={best_f1[0]:.2f}, precision={best_f1[1]:.3f}, "
        f"recall={best_f1[2]:.3f}, f1={best_f1[3]:.3f}"
    )
    print(
        f"Best Precision-> thr={best_precision[0]:.2f}, precision={best_precision[1]:.3f}, "
        f"recall={best_precision[2]:.3f}, f1={best_precision[3]:.3f}"
    )


if __name__ == "__main__":
    check_prerequisites()
    run_threshold_sweep()
