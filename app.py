from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "srcs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chatbot
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Aquaponics Expert AI",
    page_icon="🐟",
    layout="centered",
)

# ==========================================
# Minimalist Custom CSS
# ==========================================
def get_theme_css(theme: str) -> str:
    """Return CSS based on current theme."""
    if theme == "dark":
        return """
<style>
/* Dark theme colors */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .st-bp {
    color: #b0b0b0 !important;
}

.stChatMessage {
    background-color: transparent;
    border-bottom: 1px solid #30363d;
}

.stChatInputContainer {
    border-top: 1px solid #30363d;
}

.stChatInputContainer textarea {
    background-color: #0d1117;
    color: #fafafa;
}

.stMarkdown {
    color: #fafafa;
}

h1, h2, h3, h4, h5, h6 {
    color: #fafafa !important;
}

code {
    background-color: #21262d;
    color: #79c0ff;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Clean header */
header {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
"""
    else:
        return """
<style>
/* Light theme colors */
.stApp {
    background-color: #ffffff;
    color: #262730;
}

[data-testid="stSidebar"] {
    background-color: #f9f9f9;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label {
    color: #555555 !important;
}

.stChatMessage {
    background-color: transparent;
    border-bottom: 1px solid #eee;
}

.stChatInputContainer {
    border-top: 1px solid #eee;
}

.stMarkdown {
    color: #262730;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Clean header */
header {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
"""

# ==========================================
# Session State Initialization
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "original_problem" not in st.session_state:
    st.session_state.original_problem = ""
if "mode" not in st.session_state:
    st.session_state.mode = "simple"
if "follow_up_count" not in st.session_state:
    st.session_state.follow_up_count = 0
if "task_type" not in st.session_state:
    st.session_state.task_type = "simple"
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# ==========================================
# Helper Functions
# ==========================================
def get_task_type_label(task_type: str) -> str:
    """Return display label for task type."""
    labels = {
        "diagnostic": "🔍 Diagnostic",
        "design": "🏗️ Design",
        "cost": "💰 Cost Estimate",
        "simple": "Q&A",
    }
    return labels.get(task_type, "Unknown")


def clear_chat():
    """Reset chat session state."""
    st.session_state.messages = []
    st.session_state.original_problem = ""
    st.session_state.mode = "simple"
    st.session_state.follow_up_count = 0
    st.session_state.task_type = "simple"


def invoke_chatbot(user_input: str) -> str:
    """Invoke the chatbot workflow and return response."""
    current_state = {
        "messages": [
            HumanMessage(content=msg["content"])
            for msg in st.session_state.messages
            if msg["role"] in ["user", "assistant"]
        ],
        "user_query": user_input,
        "original_problem": st.session_state.original_problem,
        "mode": st.session_state.mode,
        "follow_up_count": st.session_state.follow_up_count,
        "task_type": st.session_state.task_type,
    }

    try:
        result_state = chatbot.app.invoke(current_state)

        if result_state.get("messages"):
            last_message = result_state["messages"][-1]
            st.session_state.task_type = result_state.get("task_type", "simple")
            st.session_state.mode = result_state.get("mode", "simple")
            st.session_state.follow_up_count = result_state.get("follow_up_count", 0)

            return last_message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

    return "No response generated."


# ==========================================
# Header
# ==========================================
st.title("🐟 Aquaponics Expert")
st.caption("Intelligent assistant for diagnostics, design, and advice")


# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.header("Settings")

    # Theme toggle
    theme_options = ["light", "dark"]
    current_theme_idx = theme_options.index(st.session_state.theme)
    new_theme = st.selectbox("Theme", options=theme_options, index=current_theme_idx)
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()

    # Session info
    if st.session_state.task_type != "simple":
        st.info(f"Mode: {get_task_type_label(st.session_state.task_type)}")

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()

    st.divider()

    # Examples
    st.subheader("Examples")
    st.markdown(
        """
        **Diagnostics**
        - Fish gasping at surface
        - Yellow plant leaves

        **Design**
        - Build system for tomatoes
        - Best system for rooftop?

        **Cost**
        - Cost for 100L NFT system?
        """
    )


# ==========================================
# Chat Interface
# ==========================================

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about aquaponics..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Update original problem if this is the first message
    if st.session_state.follow_up_count == 0:
        st.session_state.original_problem = prompt

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = invoke_chatbot(prompt)
            st.markdown(response)

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Show follow-up indicator if needed
    if st.session_state.mode == "complex" and st.session_state.follow_up_count > 0:
        st.info(f"💬 Please provide more details to continue...")
