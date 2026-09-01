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

from calculators.finance import (
    DEFAULT_ANNUAL_RETURN,
    asset_allocation_split,
    lean_fire_number,
    required_monthly_sip,
    sip_future_value,
)
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


def render_ask_tab() -> None:
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


def render_allocation_breakdown(total_amount: float) -> None:
    """Show a corpus split across Sharan's 60/10/15/5/5/5 asset-class framework."""
    split = asset_allocation_split(total_amount)
    st.caption("How that corpus breaks down across Sharan's asset-allocation framework:")
    st.bar_chart(split)
    for asset_class, amount in split.items():
        st.write(f"- **{asset_class}**: ₹{amount:,.0f}")


def render_fire_calculator() -> None:
    st.subheader("How much do I need to invest to hit my FIRE number?")
    st.caption(
        "Lean FIRE = annual expenses × 20 (a 5% withdrawal rate) — Sharan's own rule, "
        "not a generic finance-industry one."
    )

    monthly_expenses = st.number_input(
        "Current monthly expenses (₹)", min_value=0, value=50_000, step=5_000, key="fire_expenses"
    )
    years = st.slider("Years until you want to reach it", 1, 40, 13, key="fire_years")
    annual_return = st.slider(
        "Expected annual return (%)", 1, 20, int(DEFAULT_ANNUAL_RETURN * 100), key="fire_return"
    ) / 100

    annual_expenses = monthly_expenses * 12
    fire_number = lean_fire_number(annual_expenses)
    required_sip = required_monthly_sip(fire_number, annual_return, years)

    col1, col2 = st.columns(2)
    col1.metric("Your lean FIRE number", f"₹{fire_number:,.0f}")
    col2.metric("Required monthly SIP", f"₹{required_sip:,.0f}")

    render_allocation_breakdown(fire_number)


def render_sip_growth_calculator() -> None:
    st.subheader("What will my SIP grow to?")

    monthly_amount = st.number_input(
        "Monthly SIP amount (₹)", min_value=0, value=20_000, step=1_000, key="sip_amount"
    )
    years = st.slider("Investment horizon (years)", 1, 40, 10, key="sip_years")
    annual_return = st.slider(
        "Expected annual return (%)", 1, 20, int(DEFAULT_ANNUAL_RETURN * 100), key="sip_return"
    ) / 100

    future_value = sip_future_value(monthly_amount, annual_return, years)
    total_invested = monthly_amount * years * 12

    col1, col2 = st.columns(2)
    col1.metric("Future value", f"₹{future_value:,.0f}")
    col2.metric("Total invested", f"₹{total_invested:,.0f}", f"₹{future_value - total_invested:,.0f} growth")

    render_allocation_breakdown(future_value)


def render_calculators_tab() -> None:
    st.caption(
        "Real math, not an LLM guessing at arithmetic — see calculators/finance.py "
        "for exactly which transcript moment each number is grounded in."
    )
    render_fire_calculator()
    st.divider()
    render_sip_growth_calculator()


def main() -> None:
    st.set_page_config(page_title="1% Club Finance Assistant", page_icon="💰")
    st.title("1% Club Finance Assistant")
    st.caption("Ask questions grounded in Sharan Hegde's financial education content.")

    ask_tab, calculators_tab = st.tabs(["💬 Ask", "📊 Calculators"])
    with ask_tab:
        render_ask_tab()
    with calculators_tab:
        render_calculators_tab()


if __name__ == "__main__":
    main()
