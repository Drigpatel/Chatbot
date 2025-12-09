import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path(__file__).parent / "faiss_index.bin"
META_PATH = Path(__file__).parent / "faiss_meta.json"


class EmbeddingIndex:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []

    def build(self, questions_json_path: str):
        """Create embeddings index from questions dataset."""
        with open(questions_json_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        texts = [q["question"] for q in questions]
        self.meta = questions

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

        print("Embedding index built successfully.")

    def load(self):
        """Load an existing FAISS index from disk."""
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            raise RuntimeError("Index not found. Please run build() first.")

    def query(self, text: str, top_k: int = 5):
        """Search similar questions and return structured results."""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        vector = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vector)

        distances, indices = self.index.search(vector, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            meta_item = self.meta[idx]

            results.append({
                "score": float(score),
                "question": meta_item.get("question", ""),
                "id": meta_item.get("id", idx),
                "metadata": meta_item
            })

        return results
