# src/evaluate_system.py

import os
import re
from typing import List, Dict, Any

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util



BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
TEST_FILE = os.path.join(BASE_DIR, "data", "CleanTech-50answers.txt")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10

def parse_cleantech_50answers(path: str) -> List[Dict[str, Any]]:
    """
    Parse CleanTech-50answers.txt into list of test cases.

    Expected Format:
    **1. Query:** <question>
    **Desired Output:** <gold answer>
    **Referenced Articles:** <id1>, <id2>, ...

    Works with your exact file format.
    """

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r"\*\*\d+\. Query:\*\*", text)
    blocks = blocks[1:]  # remove header

    cases = []
    for block in blocks:

        # QUESTION
        q_match = re.search(
            r"^(.*?)\*\*Desired Output:\*\*",
            block,
            re.S
        )
        if not q_match:
            continue
        question = q_match.group(1).strip()

        # DESIRED ANSWER
        a_match = re.search(
            r"\*\*Desired Output:\*\*(.*?)\*\*Referenced Articles:\*\*",
            block,
            re.S
        )
        if not a_match:
            continue
        desired = a_match.group(1).strip()

        # REFERENCES
        r_match = re.search(r"\*\*Referenced Articles:\*\*(.*)", block)
        if not r_match:
            continue

        refs_raw = r_match.group(1).strip()
        refs = [r.strip() for r in refs_raw.split(",") if r.strip().isdigit()]

        cases.append({
            "question": question,
            "desired_answer": desired,
            "references": refs
        })

    print(f"Parsed {len(cases)} test cases from {path}")
    return cases

def load_retriever():
    print("Loading embeddings + Chroma...")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
        collection_name="cleantech_media"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    return retriever

def compute_retrieval_metrics(test_cases, retriever, k=TOP_K):

    rows = []

    for idx, case in enumerate(test_cases):
        question = case["question"]
        gold_ids = set(case["references"])

        docs = retriever.invoke(question)

        retrieved_ids = []
        for d in docs[:k]:
            doc_id = str(d.metadata.get("id", "")).strip()
            retrieved_ids.append(doc_id)

        relevant_flags = [doc_id in gold_ids for doc_id in retrieved_ids]

        num_gold = len(gold_ids)
        num_found = sum(relevant_flags)

        hit = 1.0 if num_found > 0 else 0.0
        precision = num_found / max(k, 1)
        recall = num_found / max(num_gold, 1)

        rr = 0.0
        for r, flag in enumerate(relevant_flags, start=1):
            if flag:
                rr = 1.0 / r
                break

        rows.append({
            "idx": idx,
            "question": question,
            "gold": list(gold_ids),
            "retrieved": retrieved_ids,
            "num_retrieved_relevant": num_found,
            "hit@k": hit,
            "precision@k": precision,
            "recall@k": recall,
            "reciprocal_rank": rr
        })

    return pd.DataFrame(rows)


def summarize_retrieval_metrics(df):
    return {
        "mean_hit@k": df["hit@k"].mean(),
        "mean_precision@k": df["precision@k"].mean(),
        "mean_recall@k": df["recall@k"].mean(),
        "mean_mrr": df["reciprocal_rank"].mean(),
    }

def compute_answer_metrics(test_cases, answers):

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    rows = []
    for case, ans in zip(test_cases, answers):
        gold = case["desired_answer"]

        emb_gold = model.encode(gold, convert_to_tensor=True)
        emb_ans = model.encode(ans, convert_to_tensor=True)

        cosine_sim = float(util.cos_sim(emb_gold, emb_ans)[0][0])

        rows.append({
            "question": case["question"],
            "gold_answer": gold,
            "model_answer": ans,
            "cosine_similarity": cosine_sim
        })

    return pd.DataFrame(rows)

def main():

    cases = parse_cleantech_50answers(TEST_FILE)

    retriever = load_retriever()

    print("\nEvaluating retrieval...")
    df_ret = compute_retrieval_metrics(cases, retriever, k=TOP_K)
    metrics = summarize_retrieval_metrics(df_ret)

    print("\n=== Retrieval Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_csv = os.path.join(BASE_DIR, "results_retrieval.csv")
    df_ret.to_csv(out_csv, index=False)
    print(f"\nSaved per-question retrieval metrics → {out_csv}")

    print("\nNext:")
    print("→ Call your MCP 'answer_question' tool for all 50 cases")
    print("→ Pass answers list into compute_answer_metrics()")


if __name__ == "__main__":
    main()
