"""
Layer 5: Streamlit frontend for the Sharan Hegde finance RAG assistant.

A chat-style UI over the RAG chain (chain/qa_chain.py): a text input plus a
few example questions, with each answer shown alongside its source video
citations as clickable, timestamped links.

Calls chain.qa_chain.ask() directly in-process rather than going over HTTP to
backend/main.py. That FastAPI backend still exists, is fully built, and is
Docker-tested (see Dockerfile/start.sh) — it's just not what serves this
deployed demo. Hugging Face's free Spaces tier only offers the Streamlit SDK
(no custom Dockerfile to run two processes), so the simplest reliable option
for the live deployment is to skip the HTTP hop entirely.

Run from the project root:
    ./.venv/Scripts/python.exe -m streamlit run frontend/app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Make the `chain` package importable regardless of the working directory
# Streamlit was launched from (matters both locally and once deployed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# chain.qa_chain reads GROQ_API_KEY from os.environ at import time. Locally
# that's populated by load_dotenv() reading .env. On Streamlit Community
# Cloud, secrets are set via st.secrets instead — bridge it into os.environ
# here, before the import below triggers chain.qa_chain's module-level code,
# so the same os.environ lookup works unchanged on either platform.
try:
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # no secrets.toml / no Streamlit secrets configured — fine locally

from chain.qa_chain import ask as run_qa_chain

# Not using an actual photo of Sharan Hegde here — that's a real person's
# likeness and not something to source without his permission. An emoji
# avatar gives the assistant a bit of personality without that problem.
ASSISTANT_AVATAR = "🧑‍🏫"

EXAMPLE_QUESTIONS = [
    "What are SIPs and how much of my income should go into them?",
    "How should I split my portfolio across asset classes in my 20s?",
    "How do I build an emergency fund?",
    "What's the biggest financial mistake young professionals make?",
]


def render_sources(sources: list[dict]) -> None:
    """Render a list of source citations as an expandable block of timestamped links."""
    with st.expander("Sources"):
        for source in sources:
            minutes, seconds = divmod(source["start_time"], 60)
            st.markdown(f"- [{source['title']} @ {minutes}:{seconds:02d}]({source['source_url']})")


def handle_question(question: str) -> None:
    """Record the user's question, run the RAG chain, and record + render the assistant's reply."""
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                result = run_qa_chain(question)
            except Exception as e:
                st.error(f"Something went wrong answering that: {e}")
                return
        st.markdown(result["answer"])
        render_sources(result["sources"])
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )


def main() -> None:
    st.set_page_config(page_title="1% Club Finance Assistant", page_icon="💰")
    st.title("1% Club Finance Assistant")
    st.caption("Ask questions grounded in Sharan Hegde's financial education content.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.write("Try an example:")
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked_example = None
    for col, question in zip(example_cols, EXAMPLE_QUESTIONS):
        if col.button(question):
            clicked_example = question

    # Replay prior turns on every rerun — Streamlit re-executes the whole
    # script on each interaction, so session_state is the only thing that
    # persists chat history across reruns.
    for message in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    typed_question = st.chat_input("Ask about savings, investing, taxes, insurance...")
    question = clicked_example or typed_question
    if question:
        handle_question(question)


if __name__ == "__main__":
    main()
