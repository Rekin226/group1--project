from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "srcs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chatbot  # noqa: E402


st.set_page_config(
    page_title="Aquaponics Assistant",
    page_icon="A",
    layout="wide",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-1: #f6f1ea;
  --bg-2: #e7efe9;
  --ink: #1d1c1a;
  --muted: #5a5a56;
  --accent: #2b6f5a;
  --accent-2: #8db19a;
  --bubble-user: #ffffff;
  --bubble-ai: #f0f6f3;
  --border: rgba(0,0,0,0.08);
}

html, body, [class*="stApp"] {
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
}

.stApp {
  background: radial-gradient(1200px 800px at 10% 0%, #ffffff 0%, var(--bg-1) 35%, var(--bg-2) 100%);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f2f6f1 100%);
  border-right: 1px solid var(--border);
}

.app-title {
  font-weight: 700;
  font-size: 1.4rem;
  letter-spacing: 0.02em;
  margin-bottom: 0.2rem;
}

.app-subtitle {
  color: var(--muted);
  font-size: 0.9rem;
  margin-bottom: 1rem;
}

.chip {
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 0.75rem;
  color: var(--muted);
}

.hint {
  font-size: 0.85rem;
  color: var(--muted);
}

[data-testid="stChatMessage"] {
  border-radius: 14px;
  padding: 0.4rem 0.8rem;
  border: 1px solid var(--border);
  background: var(--bubble-ai);
}

[data-testid="stChatMessage"][data-role="user"] {
  background: var(--bubble-user);
  border: 1px solid var(--border);
}

code, pre {
  font-family: "IBM Plex Mono", monospace;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def _load_vectorstore():
    chatbot.requests_cache.install_cache(chatbot.CACHE_NAME, expire_after=chatbot.CACHE_EXPIRE)
    return chatbot.build_rag_index_from_urls()


def _ensure_backend():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = _load_vectorstore()
    chatbot.VECTORSTORE = st.session_state.vectorstore


def _init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Tell me about your system or what you want to build, and I will guide you.",
            }
        ]
        chatbot.reset_state()


_ensure_backend()
_init_chat_state()


with st.sidebar:
    st.markdown('<div class="app-title">Aquaponics Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Diagnosis and design support</div>', unsafe_allow_html=True)

    if st.session_state.vectorstore is None:
        st.markdown('<div class="chip">RAG: disabled</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chip">RAG: enabled</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="hint">Ask about system design, water quality, fish behavior, or plant issues.</div>',
        unsafe_allow_html=True,
    )

    if st.button("New chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Tell me about your system or what you want to build, and I will guide you.",
            }
        ]
        chatbot.reset_state()
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Describe your system or ask a question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.handle_turn(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
