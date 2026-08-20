import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from vectorstore import build_vectorstore, retrieve

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

Rules:
- Ground every claim in the provided context. Do not invent numbers, rules, or advice
  that isn't supported by the context.
- If the context doesn't contain enough information to answer, say so plainly instead
  of guessing or falling back on generic financial knowledge.
- Keep Sharan's practical, no-nonsense tone — direct and actionable, not textbook-dry.
- This is educational content, not personalized financial advice. Don't tell the user
  what to specifically do with their own money as if you were their advisor.
"""

            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


vectorstore = build_vectorstore()


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c['title']} @ {c['start_time']}s]\n{c['text']}" for c in chunks
    )


def to_citations(chunks: list[dict]) -> list[dict]:
    return [
        {"title": c["title"], "start_time": c["start_time"], "source_url": c["source_url"]}
        for c in chunks
    ]


def qa_build_chain(retrieve):
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

if __name__ == "__main__":
    result = chain1.invoke("what are SIPs and how much percentage of capital should be invested into it?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['title']} @ {s['start_time']}s — {s['source_url']}")