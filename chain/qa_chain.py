"""
Layer 3: Retrieval-augmented QA chain over the Sharan Hegde transcript corpus.

Wires together chain/vectorstore.py's retrieval with a Groq-hosted LLM to answer
a question grounded only in retrieved transcript chunks, returning both the
generated answer and the source chunks it was built from (for citations).

Exposes `ask(question)` as the single entry point other layers should call —
backend/main.py's /ask endpoint imports this directly rather than touching the
LCEL chain internals.

Because this now lives inside the `chain` package, run it from the project root
with `-m` so the package import below resolves:
    ./.venv/Scripts/python.exe -m chain.qa_chain
"""

import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from chain.formatting import format_context, to_citations
from chain.vectorstore import build_vectorstore, retrieve

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, api_key=os.environ["GROQ_API_KEY"])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial education assistant trained exclusively on
Sharan Hegde's (founder of the 1% Club) teaching content. Answer the user's question
using ONLY the context passages provided below — they are transcript excerpts from
Sharan's videos.

Grounding rules:
- Ground every claim in the provided context. Do not invent numbers, rules, or advice
  that isn't supported by the context.
- If the context doesn't contain enough information to answer, say so plainly instead
  of guessing or falling back on generic financial knowledge.
- This is educational content, not personalized financial advice. Don't tell the user
  what to specifically do with their own money as if you were their advisor.

Voice — match Sharan's teaching style, not generic financial-advisor writing:
- Talk directly at the reader in second person ("you"), the way he talks to his
  audience — not a detached third-person explainer.
- Build points the way he does: pose the question, then answer it. ("What's leverage?
  It's...") Don't just state conclusions cold.
- Be concrete and numbers-heavy. Reach for actual rupee figures and relatable
  everyday scenarios (EMIs, subscriptions, a specific salary/expense breakdown) from
  the context instead of abstract percentages alone.
- Be blunt about hard truths when the context supports it ("a 9-to-5 alone won't get
  you there") rather than hedging everything into mush.
- Short, punchy sentences mixed in with the explanatory ones — not a wall of uniform
  textbook prose.
- Don't force in slang ("bro", "guys", "man") — that's crowd banter from a live
  seminar, not his actual teaching voice, and forcing it in reads as caricature.
"""

            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


vectorstore = build_vectorstore()


def qa_build_chain(retrieve):
    """Build the LCEL runnable: retrieve chunks once, then fan out into an answer branch and a sources branch."""
    retrieve_chunks = RunnableLambda(lambda question: retrieve(vectorstore, question))

    # First, fan out into {chunks, question}. Then fan out AGAIN from that result:
    # one branch formats the chunks + question into the LLM prompt to get the answer,
    # the other just carries the raw chunks through as citations — so retrieval only
    # happens once, but its output survives past the point where StrOutputParser would
    # otherwise throw it away.
    return (
        RunnableParallel({"chunks": retrieve_chunks, "question": RunnablePassthrough()})
        | RunnableParallel({
            "answer": (
                RunnableLambda(lambda x: {"context": format_context(x["chunks"]), "question": x["question"]})
                | prompt
                | model
                | StrOutputParser()
            ),
            "sources": RunnableLambda(lambda x: to_citations(x["chunks"])),
        })
    )


chain1 = qa_build_chain(retrieve)


def ask(question: str) -> dict:
    """Run the RAG chain end-to-end for a single question. Returns {"answer": str, "sources": list[dict]}."""
    return chain1.invoke(question)


if __name__ == "__main__":
    result = ask("what are SIPs and how much percentage of capital should be invested into it?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['title']} @ {s['start_time']}s — {s['source_url']}")