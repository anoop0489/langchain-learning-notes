"""
Tool Calling Validation — Deterministic Evaluation

This script demonstrates validating agent tool invocations without relying on an LLM.

What it does:
1. Compares actual tool calls against expected/reference calls
2. Validates three fault planes: selection, completeness, types
3. Produces structured scores for integration into evaluation pipelines

Problem it solves:
- Agents invoke external tools (database queries, API calls, etc.)
- Tool calls have deterministic structure (name, arguments, types)
- Pure code-based checks are faster and more reliable than LLM judgment
- Catches bugs like wrong tool name, missing arguments, type mismatches

The Three Fault Planes:
1. Selection Error: Agent calls get_user_info() when it should call get_customer_info()
2. Completeness Error: Agent calls tool with only {user_id} when {user_id, department} required
3. Type Error: Agent passes user_id="123" (string) when {user_id: int} expected

Prerequisites:
- Python 3.12
- No external API keys needed

Run: uv run src/test_tool_validation.py
"""


def validate_tool_invocation(actual_call: dict, expected_call: dict) -> dict:
    """
    Deterministically validate a tool invocation against expected behavior.
    
    Args:
        actual_call: {"name": str, "args": {dict of actual args}}
        expected_call: {
            "name": str,
            "args": {dict of expected args},
            "arg_types": {arg_name: type_name}
        }
    
    Returns:
        {
            "score": 0.0–1.0,
            "issues": [list of fault descriptions],
            "comment": human-readable summary
        }
    """
    issues = []
    
    # ========== FAULT PLANE 1: SELECTION ERROR ==========
    actual_tool = actual_call.get("name", "")
    expected_tool = expected_call.get("name", "")
    
    if actual_tool != expected_tool:
        issues.append({
            "type": "selection_error",
            "severity": "critical",
            "detail": f"Tool '{actual_tool}' selected but '{expected_tool}' expected"
        })
    
    # ========== FAULT PLANE 2: COMPLETENESS ERROR ==========
    expected_args = set(expected_call.get("args", {}).keys())
    actual_args = set(actual_call.get("args", {}).keys())
    
    missing_args = expected_args - actual_args
    if missing_args:
        issues.append({
            "type": "completeness_error",
            "severity": "critical",
            "detail": f"Missing required arguments: {sorted(missing_args)}"
        })
    
    extra_args = actual_args - expected_args
    if extra_args:
        issues.append({
            "type": "extra_arguments",
            "severity": "warning",
            "detail": f"Unexpected extra arguments: {sorted(extra_args)}"
        })
    
    # ========== FAULT PLANE 3: TYPE ERRORS ==========
    expected_types = expected_call.get("arg_types", {})
    
    for arg_name, expected_type_name in expected_types.items():
        if arg_name in actual_call.get("args", {}):
            actual_value = actual_call["args"][arg_name]
            actual_type = type(actual_value).__name__
            
            # Normalize type names for comparison
            expected_normalized = expected_type_name.lower()
            actual_normalized = actual_type.lower()
            
            if actual_normalized != expected_normalized:
                issues.append({
                    "type": "type_error",
                    "severity": "critical",
                    "detail": f"Argument '{arg_name}': got {actual_type}, expected {expected_type_name}"
                })
    
    # ========== SCORE CALCULATION ==========
    if not issues:
        score = 1.0
        comment = "✅ Perfect tool invocation."
    else:
        # Count critical issues for scoring
        critical_count = sum(1 for issue in issues if issue["severity"] == "critical")
        warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
        
        # Deduct more for critical issues, less for warnings
        score = max(0.0, 1.0 - (critical_count * 0.5 + warning_count * 0.1))
        
        issue_summary = " | ".join([issue["detail"] for issue in issues])
        comment = f"Issues detected: {issue_summary}"
    
    return {
        "score": round(score, 2),
        "issues": issues,
        "comment": comment
    }


def print_result(test_name: str, result: dict):
    """Pretty-print a validation result."""
    print(f"\n{'─' * 70}")
    print(f"Test: {test_name}")
    print(f"{'─' * 70}")
    print(f"Score: {result['score']}/1.0")
    print(f"Status: {'✅ PASS' if result['score'] == 1.0 else '⚠️  FAIL'}")
    
    if result['issues']:
        print(f"\nIssues ({len(result['issues'])}):")
        for issue in result['issues']:
            severity_badge = "🔴" if issue['severity'] == "critical" else "🟡"
            print(f"  {severity_badge} [{issue['type']}] {issue['detail']}")
    
    print(f"\nComment: {result['comment']}\n")


if __name__ == "__main__":
    print("=" * 70)
    print("TOOL CALLING VALIDATION — DETERMINISTIC EVALUATION")
    print("=" * 70)
    
    # ========== TEST 1: PERFECT INVOCATION ==========
    print("\n\n🧪 TEST 1: Perfect Tool Invocation")
    result = validate_tool_invocation(
        actual_call={
            "name": "fetch_employee_policy",
            "args": {
                "policy_id": "int_remote_30",
                "include_amendments": True
            }
        },
        expected_call={
            "name": "fetch_employee_policy",
            "args": {
                "policy_id": "int_remote_30",
                "include_amendments": True
            },
            "arg_types": {
                "policy_id": "str",
                "include_amendments": "bool"
            }
        }
    )
    print_result("Perfect Invocation", result)
    
    # ========== TEST 2: SELECTION ERROR ==========
    print("\n🧪 TEST 2: Wrong Tool Selected (Selection Error)")
    result = validate_tool_invocation(
        actual_call={
            "name": "fetch_wrong_policy",  # ❌ Wrong tool
            "args": {"policy_id": "int_remote_30"}
        },
        expected_call={
            "name": "fetch_employee_policy",
            "args": {"policy_id": "int_remote_30"},
            "arg_types": {"policy_id": "str"}
        }
    )
    print_result("Wrong Tool Name", result)
    
    # ========== TEST 3: MISSING ARGUMENTS ==========
    print("\n🧪 TEST 3: Missing Required Arguments (Completeness Error)")
    result = validate_tool_invocation(
        actual_call={
            "name": "search_employee_records",
            "args": {
                "employee_id": 12345
                # ❌ Missing: department_code
            }
        },
        expected_call={
            "name": "search_employee_records",
            "args": {
                "employee_id": 12345,
                "department_code": "HR"
            },
            "arg_types": {
                "employee_id": "int",
                "department_code": "str"
            }
        }
    )
    print_result("Missing Arguments", result)
    
    # ========== TEST 4: TYPE MISMATCH ==========
    print("\n🧪 TEST 4: Argument Type Mismatch (Type Error)")
    result = validate_tool_invocation(
        actual_call={
            "name": "process_request",
            "args": {
                "request_id": "REQUEST_123",  # ❌ Should be int
                "urgent": True
            }
        },
        expected_call={
            "name": "process_request",
            "args": {
                "request_id": 123,
                "urgent": True
            },
            "arg_types": {
                "request_id": "int",
                "urgent": "bool"
            }
        }
    )
    print_result("Type Mismatch", result)
    
    # ========== TEST 5: MULTIPLE ISSUES ==========
    print("\n🧪 TEST 5: Multiple Faults (Combined Errors)")
    result = validate_tool_invocation(
        actual_call={
            "name": "get_user",  # ❌ Wrong tool (selection)
            "args": {
                "user_id": "abc"  # ❌ Type mismatch (should be int)
                # ❌ Missing: department_id
            }
        },
        expected_call={
            "name": "get_employee",
            "args": {
                "user_id": 123,
                "department_id": "HR"
            },
            "arg_types": {
                "user_id": "int",
                "department_id": "str"
            }
        }
    )
    print_result("Multiple Faults", result)
    
    # ========== TEST 6: EXTRA ARGUMENTS (Warning Only) ==========
    print("\n🧪 TEST 6: Extra Unexpected Arguments (Warning, Not Critical)")
    result = validate_tool_invocation(
        actual_call={
            "name": "fetch_employee",
            "args": {
                "employee_id": 456,
                "include_salary": True,  # ✅ Correct
                "extra_field": "unexpected"  # ⚠️  Not expected
            }
        },
        expected_call={
            "name": "fetch_employee",
            "args": {
                "employee_id": 456,
                "include_salary": True
            },
            "arg_types": {
                "employee_id": "int",
                "include_salary": "bool"
            }
        }
    )
    print_result("Extra Arguments", result)
    
    print("\n" + "=" * 70)
    print("✅ Tool Validation Test Suite Complete")
    print("=" * 70 + "\n")
