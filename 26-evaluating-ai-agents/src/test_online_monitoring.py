"""
Online Hallucination Detection via Self-Consistency

This script demonstrates detecting hallucinations in production without ground truth.

What it does:
1. Takes a user question and an AI model response
2. Samples multiple variations of the response using high temperature
3. Embeds all responses and computes cosine distance variance
4. If variance exceeds threshold (0.25), flags as potential hallucination
5. Produces a score for online monitoring

Problem it solves:
- In production, you don't have human-verified answers to compare against
- LLMs can confidently generate false information (hallucinations)
- Self-consistency is a reference-free signal: if the model truly knows something,
  it will reproduce the same core meaning across high-temperature samples

Why it works:
- Factual knowledge is stable: "Paris is the capital of France" consistently paraphrases
- Hallucinations are unstable: made-up facts diverge wildly across samples
- Embedding distance captures semantic divergence better than string comparison

Prerequisites:
- OPENAI_API_KEY in .env

Run: uv run src/test_online_monitoring.py
"""

import os
import truststore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np

# Corporate proxy SSL
truststore.inject_into_ssl()

# Load environment variables
load_dotenv()


def check_prerequisites():
    """Validate required API keys."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set in .env")
    print("✅ Prerequisites validated\n")


def hallucination_detector(user_question: str, primary_output: str = None, num_samples: int = 3) -> dict:
    """
    Detect hallucination risk using self-consistency.
    
    Algorithm:
    1. Generate primary output (temperature=0, deterministic)
    2. Sample N variations with high temperature (0.8)
    3. Embed all outputs and compute pairwise cosine distances
    4. If average variance > threshold, flag as hallucination
    
    Args:
        user_question: The question/prompt
        primary_output: Optional pre-generated output; if None, generate one
        num_samples: Number of variations to sample (3–5 recommended)
    
    Returns:
        Dictionary with keys:
        - score: 0.0 (hallucination) or 1.0 (stable)
        - variance: float, average cosine distance
        - comment: Human-readable explanation
    """
    # Initialize models
    deterministic_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    sampling_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.8,  # High variability to test consistency
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    embedder = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Step 1: Generate or use provided primary output
    if primary_output is None:
        print(f"🔹 Generating primary output for: '{user_question}'")
        response = deterministic_model.invoke(user_question)
        primary_output = response.content
    
    print(f"📝 Primary output: {primary_output[:80]}...\n")
    
    # Step 2: Sample high-temperature variations
    print(f"🔄 Sampling {num_samples} high-temperature variations...")
    samples = []
    for i in range(num_samples):
        response = sampling_model.invoke(user_question)
        samples.append(response.content)
        print(f"   Sample {i+1}: {response.content[:60]}...")
    
    # Step 3: Embed and compute distances
    print("\n📊 Computing semantic variance...")
    primary_vec = embedder.embed_query(primary_output)
    sample_vecs = embedder.embed_documents(samples)
    
    distances = []
    for i, vec in enumerate(sample_vecs):
        # Cosine similarity
        cos_sim = np.dot(primary_vec, vec) / (
            np.linalg.norm(primary_vec) * np.linalg.norm(vec)
        )
        # Cosine distance (1 - similarity)
        cos_dist = 1.0 - cos_sim
        distances.append(cos_dist)
        print(f"   Distance to sample {i+1}: {cos_dist:.3f}")
    
    avg_variance = sum(distances) / len(distances)
    print(f"\n📈 Average variance: {avg_variance:.3f}")
    
    # Step 4: Decision threshold
    HALLUCINATION_THRESHOLD = 0.25  # Tune based on your production data
    
    if avg_variance > HALLUCINATION_THRESHOLD:
        print(f"\n⚠️  HALLUCINATION ALERT: Variance {avg_variance:.3f} exceeds threshold {HALLUCINATION_THRESHOLD}")
        return {
            "key": "hallucination_risk",
            "score": 0.0,
            "variance": round(avg_variance, 3),
            "comment": f"High internal divergence ({avg_variance:.3f}). Model may be hallucinating."
        }
    
    print(f"\n✅ STABLE OUTPUT: Variance {avg_variance:.3f} is within safe range")
    return {
        "key": "hallucination_risk",
        "score": 1.0,
        "variance": round(avg_variance, 3),
        "comment": f"Output is internally consistent (variance {avg_variance:.3f})."
    }


if __name__ == "__main__":
    check_prerequisites()
    
    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "description": "Factual, well-known fact (low hallucination risk)"
        },
        {
            "question": "Explain quantum entanglement in one sentence.",
            "description": "Technical but well-documented (medium hallucination risk)"
        },
        {
            "question": "What did the CEO announce in last month's earnings call? (No context provided)",
            "description": "Without context, likely to hallucinate (high hallucination risk)"
        }
    ]
    
    print("=" * 80)
    print("ONLINE HALLUCINATION DETECTION — SELF-CONSISTENCY MONITORING")
    print("=" * 80 + "\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}: {test_case['description']}")
        print(f"{'=' * 80}\n")
        
        print(f"❓ Question: {test_case['question']}\n")
        
        result = hallucination_detector(test_case["question"], num_samples=3)
        
        print(f"\n📋 Result:")
        print(f"   Hallucination Risk Score: {result['score']}")
        print(f"   Variance: {result['variance']}")
        print(f"   Assessment: {result['comment']}\n")
    
    print("=" * 80)
    print("✅ Demonstration complete")
    print("=" * 80 + "\n")
