import json
from pathlib import Path
from math import sqrt
from openai import OpenAI
import os

META_PATH = Path(__file__).parent / "faiss_meta.json"

class EmbeddingIndex:
    def __init__(self):
        self.client = OpenAI()
        self.meta = []
    
    def build(self, questions_json_path):
        with open(questions_json_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        
        with open(META_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.meta, f)

    def load(self):
        if META_PATH.exists():
            with open(META_PATH, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
        else:
            raise RuntimeError("Index not built. Call build() first.")

    def embed(self, text):
        res = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return res.data[0].embedding

    def cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sqrt(sum(x * x for x in a))
        mag_b = sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b)

    def query(self, text, top_k=5):
        q_emb = self.embed(text)
        results = []

        for item in self.meta:
            emb = self.embed(item["question"])
            score = self.cosine(q_emb, emb)
            results.append({"score": score, "meta": item})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
