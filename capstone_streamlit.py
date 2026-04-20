"""
capstone_streamlit.py — First-Aid & Emergency Health FAQ Bot
All agent logic lives in agent.py — this file is UI only.
Run: streamlit run capstone_streamlit.py
"""

import uuid
import streamlit as st
from agent_1 import build_agent, DOCUMENTS

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚑 First-Aid Bot",
    page_icon="🚑",
    layout="centered",
)

st.title("🚑 First-Aid & Emergency Health FAQ Bot")
st.caption("Step-by-step emergency guidance — CPR, burns, choking, strokes and more.")


# ──────────────────────────────────────────────────────────────
# AGENT INITIALISATION  — cached so it never reloads on rerun
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    """Load LLM, embedder, ChromaDB, and compiled graph exactly once."""
    return build_agent()


try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents ready")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()


# ──────────────────────────────────────────────────────────────
# SESSION STATE  — messages and thread_id reset on new conversation
# ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🚑 About")
    st.write(
        "This bot provides step-by-step first-aid guidance for common emergencies. "
        "It uses a retrieval-augmented LangGraph agent with memory."
    )
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    st.divider()

    st.write("**Topics covered:**")
    for d in DOCUMENTS:
        st.write(f"• {d['topic']}")

    st.divider()

    if st.button("🔄 New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.warning(
        "⚠️ This bot provides general first-aid guidance only. "
        "Always call **112** in a real emergency."
    )


# ──────────────────────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ──────────────────────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a first-aid question..."):

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Looking up first-aid guidance..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")

        st.write(answer)

        # Debug metadata
        faith = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        route = result.get("route", "?")
        if faith > 0:
            st.caption(
                f"Faithfulness: {faith:.2f} | Route: {route} | Sources: {sources}"
            )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.info("🚨 For real emergencies, always call **112** immediately.")
