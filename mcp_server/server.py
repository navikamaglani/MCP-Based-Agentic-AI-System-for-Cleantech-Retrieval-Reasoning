# mcp_server/server.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

from fastmcp import FastMCP

# Vector DB + Embeddings
from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient
from sentence_transformers import CrossEncoder
mcp = FastMCP("cleantech-mcp")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"

embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedder,
    collection_name="cleantech_media",
)

retriever = vectordb.as_retriever(search_kwargs={"k": 20})

# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM
hf_client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct:free")


def run_llm(prompt: str) -> str:
    try:
        out = hf_client.text_generation(
            prompt=prompt,
            max_new_tokens=350,
            temperature=0.2,
            do_sample=False
        )
        return out.strip()
    except Exception as e:
        return f"LLM_ERROR: {e}"

def _filter_docs(docs, domain=None, after=None, before=None):
    out = []
    for d in docs:
        md = d.metadata or {}
        if domain and domain.lower() not in str(md.get("domain", "")).lower():
            continue
        date = md.get("date")
        if date:
            if after and date < after:
                continue
            if before and date > before:
                continue
        out.append(d)
    return out


def _rerank(query, docs):
    if not docs:
        return []
    scores = reranker.predict([(query, d.page_content) for d in docs])
    combined = []
    for d, s in zip(docs, scores):
        combined.append({
            "content": d.page_content,
            "metadata": d.metadata,
            "score": float(s),
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined


def _chunk(docs, limit=6000):
    out = []
    used = 0
    for i, d in enumerate(docs):
        text = d["content"][:1000]
        block = f"[Doc {i+1}] {text}"
        if used + len(block) > limit:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


def _fallback(q, docs):
    if not docs:
        return f"Q: {q}\nNo supporting documents available."
    lines = [f"Q: {q}", "Key notes:"]
    for i, d in enumerate(docs[:5]):
        lines.append(f"- Doc{i+1}: {d['content'][:200]}...")
    return "\n".join(lines)

@mcp.tool()
def ping():
    """Simple health check."""
    return {"status": "Server OK"}


@mcp.tool()
def retrieve_cleantech(query: str,
                       k: int = 8,
                       domain: Optional[str] = None,
                       after: Optional[str] = None,
                       before: Optional[str] = None):
    """
    Retrieve cleantech documents, apply metadata filtering and reranking.
    """
    docs = retriever.invoke(query)
    docs = _filter_docs(docs, domain, after, before)
    ranked = _rerank(query, docs)[:k]
    return {
        "query": query,
        "num_results": len(ranked),
        "results": ranked,
    }


@mcp.tool()
def summarize_docs(question: str, docs: List[Dict[str, Any]]):
    """
    Summarize and answer a question using the provided documents.
    """
    if not docs:
        return {"answer": "No documents found."}

    context = _chunk(docs)

    prompt = f"""
You are a cleantech research assistant.
Use ONLY the documents below.

QUESTION:
{question}

DOCUMENTS:
{context}

Answer factually in 1–3 concise paragraphs.
"""

    out = run_llm(prompt)
    if out.startswith("LLM_ERROR"):
        out = _fallback(question, docs)

    return {"answer": out}


@mcp.tool()
def explain_answer(question: str,
                   answer: str,
                   docs: Optional[List[Dict[str, Any]]] = None):
    """
    Reasoning tool:
    Given a question, a draft answer, and optional supporting docs,
    produce a step-by-step explanation of how the answer is supported.
    """
    docs = docs or []
    context = _chunk(docs) if docs else "No documents provided."

    prompt = f"""
You are a cleantech domain expert.

We have a QUESTION, an ANSWER drafted by another model,
and some SUPPORTING DOCUMENT SNIPPETS.

Your job is to:
1. Explain, step by step, how the answer is supported by the documents.
2. Point to specific doc snippets conceptually (e.g., "Doc1 suggests that...").
3. Flag any parts of the answer that are NOT clearly supported.

QUESTION:
{question}

DRAFT ANSWER:
{answer}

SUPPORTING DOCUMENTS:
{context}

Now write a clear reasoning trace:
- Start with a short one-line verdict ("Overall, the answer is well-supported / partially supported / weakly supported").
- Then give bullet-point steps that reference the docs.
"""

    out = run_llm(prompt)
    if out.startswith("LLM_ERROR"):
        out = _fallback(question, docs)

    return {
        "question": question,
        "answer": answer,
        "reasoning": out,
    }


@mcp.tool()
def answer_question(question: str,
                    k: int = 8,
                    domain: Optional[str] = None,
                    after: Optional[str] = None,
                    before: Optional[str] = None):
    """
    Convenience tool: retrieve → summarize → explain.
    Returns both the answer and a reasoning trace.
    """
    retrieval = retrieve_cleantech(
        query=question, k=k, domain=domain, after=after, before=before
    )

    docs = retrieval["results"]
    summary_res = summarize_docs(question, docs)
    answer = summary_res["answer"]

    reasoning_res = explain_answer(question, answer, docs)

    return {
        "answer": answer,
        "docs_used": len(docs),
        "reasoning": reasoning_res["reasoning"],
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
