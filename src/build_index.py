import os
import pandas as pd
from langchain_core.documents import Document

# NEW recommended imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma   # new correct API

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "cleantech_media_dataset_v3_2024-10-28.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

BATCH_SIZE = 3000   # <= must be < 5461 to avoid Chroma crash


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    # Create stable unique IDs
    df["id"] = df["Unnamed: 0"].astype(str)

    docs = []
    for _, row in df.iterrows():
        text = str(row["content"]).strip()
        if not text or text == "nan":
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "id": row["id"],
                    "title": row["title"],
                    "url": row["url"],
                    "date": row["date"],
                    "author": row["author"],
                    "domain": row["domain"]
                }
            )
        )

    print(f"Loaded {len(docs)} documents.")

    # Embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # NEW Chroma API
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name="cleantech_media",
        embedding_function=embedder,
    )

    print("Adding documents in batches...")

    for start in range(0, len(docs), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = docs[start:end]
        print(f" → Batch {start}:{end}")
        vectordb.add_documents(batch)

   
    print(f"Indexing complete! Saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
