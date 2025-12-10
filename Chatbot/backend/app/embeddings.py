import json
from pathlib import Path
from math import sqrt
from openai import OpenAI
import os

META_PATH = Path(__file__).parent / "faiss_meta.json"
EMB_PATH = Path(__file__).parent / "question_embeddings.json"


class EmbeddingIndex:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        self.meta = []
        self.embeddings = {}

    # -----------------------------------------------------
    # Build metadata + store static embeddings only once
    # -----------------------------------------------------
    def build(self, questions_json_path):
        with open(questions_json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # Precompute embeddings ONCE — saves time and RAM
        all_embs = {}
        for item in self.meta:
            q = item["question"]
            emb = self.embed(q)
            all_embs[q] = emb

        # Save metadata
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

        # Save embeddings
        with open(EMB_PATH, "w", encoding="utf-8") as f:
            json.dump(all_embs, f)

        print("Embedding index built successfully.")

    # -----------------------------------------------------
    # Load metadata and embeddings (small JSON)
    # -----------------------------------------------------
    def load(self):
        if META_PATH.exists():
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        if EMB_PATH.exists():
            with open(EMB_PATH, "r", encoding="utf-8") as f:
                self.embeddings = json.load(f)

        if not self.meta or not self.embeddings:
            raise RuntimeError("Index not built. Run build() first.")

    # -----------------------------------------------------
    # Embed a single text using OpenAI
    # -----------------------------------------------------
    def embed(self, text):
        try:
            res = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return res.data[0].embedding
        except Exception as e:
            print("Embedding error:", e)
            return [0.0] * 1536  # fallback vector

    # -----------------------------------------------------
    # Cosine similarity
    # -----------------------------------------------------
    def cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sqrt(sum(x * x for x in a))
        mag_b = sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0
        return dot / (mag_a * mag_b)

    # -----------------------------------------------------
    # Query embeddings
    # -----------------------------------------------------
    def query(self, text, top_k=5):
        if not self.meta or not self.embeddings:
            raise RuntimeError("Embeddings not loaded. Build index first.")

        q_emb = self.embed(text)
        results = []

        for item in self.meta:
            q = item["question"]
            emb = self.embeddings.get(q)

            if emb is None:
                continue

            score = self.cosine(q_emb, emb)
            results.append({"score": score, "meta": item})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
