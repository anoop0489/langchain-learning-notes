# =============================================================================
# EDEN'S ORIGINAL: Streamlit Chat UI (main.py)
# =============================================================================
# This is Eden Marco's original Streamlit frontend — kept as-is for reference.
# Compare with our adapted version at: ../main.py
#
# HOW STREAMLIT WORKS:
#   Streamlit reruns this ENTIRE script top-to-bottom on every user interaction.
#   - Local variables RESET every rerun (like a stateless HTTP request)
#   - Only st.session_state PERSISTS between reruns (like ASP.NET Session)
#   - The message history loop re-renders ALL previous messages each time
#
# KEY STREAMLIT CONCEPTS USED:
#   - st.session_state: Per-user dict that survives reruns (chat memory)
#   - st.chat_message(): Renders a message bubble ("user" or "assistant")
#   - st.chat_input(): Fixed text box at bottom (returns None until submit)
#   - st.expander(): Collapsible section (used for source citations)
#   - st.spinner(): Loading indicator while waiting for LLM response
#   - st.rerun(): Force a fresh top-to-bottom rerun (used by Clear Chat)
#
# DIFFERENCES FROM OUR ADAPTED VERSION (../main.py):
#   - No truststore.inject_into_ssl() in this file (backend/core.py handles it)
#   - No sys.path manipulation (run from the eden-original directory)
#   - No header docblock or inline comments
#   - Imports backend.core directly (assumes cwd is project root)
#
# RUN: cd 10-documentation-assistant/src/eden-original
#      uv run streamlit run main.py
# =============================================================================

from typing import Any, Dict, List

import streamlit as st

# Import the agentic RAG backend (run_llm handles retrieval + generation)
from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """Extract source URLs from Document objects for display.

    Uses a walrus operator (:=) to extract metadata in a list comprehension.
    Each Document has .metadata dict with a "source" key containing the URL.
    Deduplicates URLs using dict.fromkeys() which preserves insertion order.
    """
    all_sources = [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]
    return list(dict.fromkeys(all_sources))


# Page config — MUST be the first st.* call in the script
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

# Sidebar with a "Clear chat" button that wipes session state and forces rerun
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

# Initialize chat history on first load — without this, messages vanish on rerun
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I'll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

# Re-render all previous messages (runs on EVERY rerun)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show source citations in a collapsible expander
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# chat_input returns None until user submits, then returns the typed text
prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
    # 1. Add user message to session state and display it immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the agentic RAG backend and display the response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                # run_llm() returns {"answer": str, "context": [Document, ...]}
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            # 3. Persist assistant response in session state for next rerun
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
