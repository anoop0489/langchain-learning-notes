"""
Tool guardrails test harness.

What this script does:
- Validates tool allowlist and argument rules.
- Demonstrates fail-closed behavior for risky tools.

Run:
uv run 27-guardrails/src/test_tool_guardrails.py
"""

import truststore
from dotenv import load_dotenv

truststore.inject_into_ssl()
load_dotenv()


def check_prerequisites():
    return True


def evaluate_tool_call(tool_name: str, args: dict) -> tuple[bool, str]:
    normalized = tool_name.strip().lower().replace("-", "_")

    allowed_tools = {"lookup_ticket", "send_email"}
    if normalized not in allowed_tools:
        return False, f"Blocked unknown tool by policy: {tool_name}"

    blocked_tools = {"send_email", "wire_transfer"}
    if normalized in blocked_tools:
        return False, f"Blocked high-risk tool by policy: {tool_name}"

    if normalized == "lookup_ticket":
        ticket_id = args.get("ticket_id", "")
        if not isinstance(ticket_id, str) or not ticket_id.startswith("TCK-"):
            return False, "Blocked: ticket_id must be string with TCK- prefix"

    return True, "Allowed"


def run_tests():
    check_prerequisites()

    samples = [
        ("lookup_ticket", {"ticket_id": "TCK-2222"}),
        ("lookup_ticket", {"ticket_id": 2222}),
        ("send_email", {"to": "ops@example.com", "body": "hello"}),
        ("sendemailnow", {"to": "ops@example.com", "body": "hello"}),
    ]

    print("=" * 70)
    print("TOOL GUARDRAILS TESTS")
    print("=" * 70)

    for idx, (tool_name, args) in enumerate(samples, start=1):
        allowed, reason = evaluate_tool_call(tool_name, args)
        print(f"\nTest {idx}")
        print(f"Tool: {tool_name}")
        print(f"Args: {args}")
        print(f"Allowed: {allowed}")
        print(f"Reason: {reason}")


if __name__ == "__main__":
    run_tests()
