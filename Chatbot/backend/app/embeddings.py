import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path(__file__).parent / "faiss_index.bin"
META_PATH = Path(__file__).parent / "faiss_meta.json"

class EmbeddingIndex:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = []

    def build(self, path):
        data = json.load(open(path, "r", encoding="utf-8"))
        texts = [item["question"] for item in data]
        self.meta = data

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, str(INDEX_PATH))
        json.dump(self.meta, open(META_PATH, "w", encoding="utf-8"))

    def load(self):
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.meta = json.load(open(META_PATH, "r", encoding="utf-8"))
        else:
            raise RuntimeError("No FAISS index found. Build first!")

    def query(self, text, top_k=5):
        if self.index is None:
            raise RuntimeError("Index not loaded")

        vec = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(vec)

        D, I = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            results.append({
                "score": float(score),
                "question": self.meta[idx]["question"],
                "id": self.meta[idx].get("id", idx),
            })
        return results
