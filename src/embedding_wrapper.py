# src/embedding_wrapper.py

from sentence_transformers import SentenceTransformer

class STEmbeddingWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # Returns list of vectors
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, query: str):
        # Returns a single vector
        return self.model.encode(query).tolist()
