# Math Question Refinement Chatbot

Minimal project: FastAPI backend + LangChain LLM flows (validation + refinement) + FAISS similarity using sentence-transformers.

Setup:
1. Install Python 3.11+
2. Create a virtualenv and pip install -r backend/requirements.txt
3. Set OPENAI_API_KEY env var and optionally OPENAI_MODEL
4. From backend: `uvicorn app.main:app --reload --port 8000`
5. Open frontend/index.html in a browser (or serve it)

Note: Building the sentence-transformers model and FAISS index happens on startup if not present. For production, prebuild the index.
