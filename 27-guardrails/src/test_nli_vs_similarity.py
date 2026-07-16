"""
NLI vs Semantic Similarity comparison for guardrails.

What this script does:
- Computes semantic similarity between two texts using embeddings.
- Runs an NLI-style directional policy check on the same text pair.
- Shows why similarity and NLI are related but not interchangeable.

Problem it solves:
- Teams often treat semantic similarity as a policy guardrail signal.
- Similarity can be high even when policy meaning is opposite.
- NLI-style checks improve allow/block decisions for nuanced rules.

Prerequisites:
- OPENAI_API_KEY

Run:
uv run 27-guardrails/src/test_nli_vs_similarity.py
"""

import json
import os
from typing import Any

import truststore
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

truststore.inject_into_ssl()
load_dotenv()


def check_prerequisites():
    required = ["OPENAI_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(text_a: str, text_b: str) -> float:
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    vec_a = embedder.embed_query(text_a)
    vec_b = embedder.embed_query(text_b)
    return cosine_similarity(vec_a, vec_b)


def nli_guardrail_decision(premise: str, hypothesis: str) -> dict[str, Any]:
    model = init_chat_model("openai:gpt-4o")

    prompt = (
        "You are a strict Natural Language Inference classifier for safety policies. "
        "Given a premise and hypothesis, return JSON only with fields: "
        "label (entailment|contradiction|neutral), confidence (0 to 1), reason (short)."
        "\n\nPremise:\n"
        f"{premise}"
        "\n\nHypothesis:\n"
        f"{hypothesis}"
    )

    result = model.invoke([{"role": "user", "content": prompt}])
    text = str(result.content).strip()

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"label": "neutral", "confidence": 0.0, "reason": f"Unparseable response: {text}"}

    label = str(parsed.get("label", "neutral")).lower()
    confidence = float(parsed.get("confidence", 0.0))
    reason = str(parsed.get("reason", ""))

    if label == "entailment" and confidence >= 0.85:
        action = "block"
    elif label == "entailment" and confidence >= 0.60:
        action = "review"
    else:
        action = "allow"

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "reason": reason,
        "action": action,
    }


def run_demo():
    check_prerequisites()

    print("=" * 80)
    print("NLI VS SEMANTIC SIMILARITY FOR GUARDRAILS")
    print("=" * 80)

    premise = "You should not take this medication daily without physician approval."
    comparison_text = "You should take this medication daily."
    hypothesis = "The text recommends daily medication use."

    sim = semantic_similarity(premise, comparison_text)
    nli = nli_guardrail_decision(premise, hypothesis)

    print("\nCase 1: High lexical overlap, opposite policy meaning")
    print(f"Premise: {premise}")
    print(f"Comparison Text: {comparison_text}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Semantic Similarity: {sim:.3f}")
    print(f"NLI Label: {nli['label']}")
    print(f"NLI Confidence: {nli['confidence']}")
    print(f"Guardrail Action: {nli['action']}")
    print(f"Reason: {nli['reason']}")

    premise_2 = "The response contains no request for passwords or API keys."
    comparison_text_2 = "No credentials are being requested in this response."
    hypothesis_2 = "The text requests credential disclosure."

    sim_2 = semantic_similarity(premise_2, comparison_text_2)
    nli_2 = nli_guardrail_decision(premise_2, hypothesis_2)

    print("\nCase 2: Similar meaning, safe outcome")
    print(f"Premise: {premise_2}")
    print(f"Comparison Text: {comparison_text_2}")
    print(f"Hypothesis: {hypothesis_2}")
    print(f"Semantic Similarity: {sim_2:.3f}")
    print(f"NLI Label: {nli_2['label']}")
    print(f"NLI Confidence: {nli_2['confidence']}")
    print(f"Guardrail Action: {nli_2['action']}")
    print(f"Reason: {nli_2['reason']}")


if __name__ == "__main__":
    run_demo()
