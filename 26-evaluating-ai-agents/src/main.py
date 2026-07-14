"""
Offline Evaluation Pipeline for Compliance Chatbot

This script demonstrates a complete offline evaluation pipeline using LangSmith.

What it does:
1. Creates a golden dataset of compliance Q&A pairs in LangSmith
2. Defines a target compliance chatbot function
3. Runs offline evaluations using LLM-as-Judge and custom guardrails
4. Logs results to LangSmith dashboard for inspection and comparison

Problem it solves:
- Deterministic testing fails for LLMs (same input → different text outputs)
- Need semantic scoring instead of string comparison
- Need to prevent PII leakage and validate outputs before production

Prerequisites:
- OPENAI_API_KEY in .env
- LANGSMITH_API_KEY in .env
- Compliance QA examples to test against

Run: uv run src/main.py
"""

import os
import truststore
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI

# Corporate proxy: inject Windows certificate store into SSL
truststore.inject_into_ssl()

# Load environment variables
load_dotenv()


def check_prerequisites():
    """Validate required environment variables before proceeding."""
    required_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}\n"
            f"Please add them to your .env file or set them as system variables."
        )
    
    print("✅ All prerequisites validated\n")


def create_or_get_golden_dataset():
    """
    Create or retrieve the golden dataset in LangSmith.
    
    The golden dataset contains:
    - Inputs: Real compliance questions
    - Outputs: Human-verified correct answers (ground truth)
    
    Returns the dataset object for use in evaluation.
    """
    client = Client()
    dataset_name = "ComplianceQA_v1"
    
    # Try to get existing dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✅ Retrieved existing dataset: {dataset_name}\n")
        return dataset
    except Exception:
        pass
    
    # Create new dataset if it doesn't exist
    print(f"📝 Creating new golden dataset: {dataset_name}\n")
    
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Production reference dataset for compliance chatbot evaluation"
    )
    
    # Define examples with human-verified ground truth
    examples = [
        {
            "input": {"question": "What is the policy on international remote work?"},
            "output": {
                "answer": "Employees may work internationally up to 30 calendar days per year with prior manager approval.",
                "reference": "Employees may work internationally up to 30 calendar days per year with prior manager approval."
            }
        },
        {
            "input": {"question": "What happens if an expense report is submitted late?"},
            "output": {
                "answer": "Late expense submissions are deferred to the next bi-weekly payroll processing cycle.",
                "reference": "Late expense submissions are deferred to the next bi-weekly payroll processing cycle."
            }
        },
        {
            "input": {"question": "What is the refund policy?"},
            "output": {
                "answer": "Refunds are allowed within 30 days of purchase with original receipt.",
                "reference": "Refunds are allowed within 30 days of purchase with original receipt."
            }
        },
        {
            "input": {"question": "How many vacation days do full-time employees get per year?"},
            "output": {
                "answer": "Full-time employees are entitled to 20 paid vacation days per calendar year.",
                "reference": "Full-time employees are entitled to 20 paid vacation days per calendar year."
            }
        }
    ]
    
    # Add examples to the dataset
    for example in examples:
        client.create_example(
            inputs=example["input"],
            outputs=example["output"],
            dataset_id=dataset.id
        )
    
    print(f"✅ Created dataset with {len(examples)} golden examples\n")
    return dataset


def compliance_chatbot(inputs: dict) -> dict:
    """
    Target application: a compliance chatbot that answers policy questions.
    
    This is the system under test. In production, you'd use your actual chatbot logic.
    
    Args:
        inputs: Dictionary with "question" key
    
    Returns:
        Dictionary with "answer" key containing the chatbot response
    """
    question = inputs.get("question", "")
    
    # Create LLM for answering questions
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,  # Deterministic for offline evaluation
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    system_prompt = """You are a compliance expert. Answer policy questions concisely and factually.
    - Base your answers only on company policy.
    - If you don't know the answer, say so rather than guessing.
    - Keep responses to 1–2 sentences."""
    
    response = model.invoke(system_prompt + f"\n\nQuestion: {question}")
    
    return {"answer": response.content}


def security_gate(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Deterministic security guardrail: fail if output contains PII or sensitive data.
    
    Checks for:
    - Social Security Numbers (SSN)
    - API keys and credentials
    - Passwords
    - Credit card numbers
    """
    import re
    
    response = outputs.get("answer", "")
    
    # Define patterns for sensitive data
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "api_key": r"(sk_|sk_test_|sk_live_)[a-zA-Z0-9]{20,}",
        "password": r"(?i)password\s*=\s*['\"]?\w+['\"]?",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
    }
    
    found_issues = []
    for issue_type, pattern in patterns.items():
        if re.search(pattern, response):
            found_issues.append(issue_type)
    
    if found_issues:
        return {
            "key": "security_check",
            "score": 0.0,
            "comment": f"🚨 CRITICAL: Detected {', '.join(found_issues)} in output."
        }
    
    return {
        "key": "security_check",
        "score": 1.0,
        "comment": "✅ No sensitive data detected."
    }


def token_efficiency_check(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Check that the answer is reasonably concise (not generating too many tokens).
    
    For compliance answers, we expect 1–2 sentences, not long essays.
    """
    response = outputs.get("answer", "")
    word_count = len(response.split())
    
    # Compliance answers should be 20–100 words
    if word_count < 10:
        return {
            "key": "conciseness",
            "score": 0.5,
            "comment": f"⚠️  Answer too short ({word_count} words). May lack detail."
        }
    elif word_count > 150:
        return {
            "key": "conciseness",
            "score": 0.5,
            "comment": f"⚠️  Answer too long ({word_count} words). Should be concise."
        }
    
    return {
        "key": "conciseness",
        "score": 1.0,
        "comment": f"✅ Appropriate length ({word_count} words)."
    }


def run_offline_evaluation():
    """Execute the complete offline evaluation pipeline."""
    print("=" * 80)
    print("OFFLINE EVALUATION PIPELINE — COMPLIANCE CHATBOT")
    print("=" * 80 + "\n")
    
    # Step 1: Validate prerequisites
    check_prerequisites()
    
    # Step 2: Initialize golden dataset
    create_or_get_golden_dataset()
    
    # Step 3: Run evaluation suite
    print("📊 Running evaluation suite...\n")
    print("Evaluators:")
    print("  - correctness (LLM-as-Judge: semantic evaluation)")
    print("  - security_gate (deterministic: PII detection)")
    print("  - token_efficiency_check (deterministic: answer length)\n")
    
    try:
        evaluate(
            compliance_chatbot,
            data="ComplianceQA_v1",
            evaluators=[
                "correctness",              # Built-in LLM-as-Judge
                security_gate,              # Custom PII check
                token_efficiency_check      # Custom length check
            ],
            experiment_prefix="offline-compliance-test"
        )
        
        print("\n" + "=" * 80)
        print("✅ EVALUATION COMPLETE")
        print("=" * 80)
        print("\n📊 Results available in LangSmith dashboard:")
        print("   https://smith.langchain.com/")
        print("\nView your experiment under 'offline-compliance-test' prefix.\n")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}\n")
        raise


if __name__ == "__main__":
    run_offline_evaluation()
