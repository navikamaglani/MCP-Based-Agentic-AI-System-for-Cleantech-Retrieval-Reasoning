# src/run_full_evaluation.py

import os
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import CrossEncoder
from huggingface_hub import InferenceClient

from src.evaluate_system import (
    parse_cleantech_50answers,
    load_retriever,
    compute_retrieval_metrics,
    summarize_retrieval_metrics,
    compute_answer_metrics,
)



BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_FILE = os.path.join(BASE_DIR, "data", "CleanTech-50answers.txt")

# same models as server
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  



def run_llm(prompt: str) -> str:
    client = InferenceClient(HF_MODEL)
    try:
        out = client.text_generation(
            prompt=prompt,
            max_new_tokens=350,
            temperature=0.2,
            do_sample=False,
        )
        return out.strip()
    except Exception as e:
        return f"LLM_ERROR: {e}"


def _rerank(query: str, docs, reranker: CrossEncoder) -> List[Dict[str, Any]]:
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


def _chunk(docs: List[Dict[str, Any]], limit: int = 6000) -> str:
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


def _fallback(question: str, docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return f"Q: {question}\nNo supporting documents available."
    lines = [f"Q: {question}", "Key notes:"]
    for i, d in enumerate(docs[:5]):
        lines.append(f"- Doc{i+1}: {d['content'][:200]}...")
    return "\n".join(lines)


def generate_answer(question: str, retriever, reranker: CrossEncoder) -> str:
    # retrieve
    docs = retriever.invoke(question)

    # rerank
    ranked = _rerank(question, docs, reranker)

    if not ranked:
        return "No documents found."

    context = _chunk(ranked)

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
        out = _fallback(question, ranked)

    return out


def main():
    print("Parsing test cases from:", TEST_FILE)
    cases = parse_cleantech_50answers(TEST_FILE)

    print("\nLoading retriever (Chroma + MiniLM)...")
    retriever = load_retriever()

    print("Loading reranker:", RERANK_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)   # ✔ FIXED
    print("\nEvaluating retrieval...")
    df_ret = compute_retrieval_metrics(cases, retriever)
    metrics = summarize_retrieval_metrics(df_ret)

    print("\n=== Retrieval Metrics (CleanTech-50answers) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    ret_path = os.path.join(BASE_DIR, "results_retrieval.csv")
    df_ret.to_csv(ret_path, index=False)
    print(f"\nSaved per-question retrieval metrics → {ret_path}")

    print("\nGenerating model answers for all test cases...")
    answers = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        print(f"  → ({i}/{len(cases)}) {q[:60]}...")
        ans = generate_answer(q, retriever, reranker)
        answers.append(ans)

    print("\nComputing answer metrics (cosine similarity)...")
    df_ans = compute_answer_metrics(cases, answers)

    mean_cos = df_ans["cosine_similarity"].mean()
    print(f"\n=== Answer Quality Metrics ===")
    print(f"Mean cosine similarity: {mean_cos:.4f}")

    ans_path = os.path.join(BASE_DIR, "results_answers.csv")
    df_ans.to_csv(ans_path, index=False)
    print(f"\nSaved per-question answer metrics → {ans_path}")
    print("\nSample rows:")
    print(df_ans[["question", "cosine_similarity"]].head(5))


if __name__ == "__main__":
    main()
