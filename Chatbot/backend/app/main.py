from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv

from app.langchain_flow import LangChainFlows
from app.embeddings import EmbeddingIndex

# Load env variables
load_dotenv()

app = FastAPI(title="ChatBot Backend API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

# OpenAI keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is missing — responses may fail.")

# Initialize LangChainFlows
flows = LangChainFlows(
    openai_api_key=OPENAI_API_KEY,
    model_name=OPENAI_MODEL
)

# Embedding index
QUESTIONS_FILE = BASE_DIR / "questions.json"
EMB = EmbeddingIndex()

try:
    EMB.load()
    print("Embeddings loaded successfully.")
except Exception:
    print("No embedding index found — building one...")
    EMB.build(str(QUESTIONS_FILE))


# -------------------- MODELS ------------------------------
class ChatRequest(BaseModel):
    message: str

class RefineRequest(BaseModel):
    question: str
    feedback: str | None = ""


# -------------------- ROUTES ------------------------------
@app.get("/")
async def root():
    return {"status": "Chatbot Backend Running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Validate the question."""
    try:
        validation = flows.validate(req.message)
        return {"validation": validation}
    except Exception as e:
        print("Chat error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine")
async def refine(req: RefineRequest):
    """Refine the question using GPT."""
    try:
        result = flows.refine(req.question, req.feedback or "")
        return {"refined_answer": result}
    except Exception as e:
        print("Refine error:", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/similarity")
async def similarity(req: RefineRequest):
    """Return similar questions based on embeddings."""
    try:
        threshold = float(os.getenv("SIM_THRESH", 0.8))
        results = EMB.query(req.question, top_k=5)

        similar = [r for r in results if r["score"] >= threshold]
        return {"similar": similar}

    except Exception as e:
        print("Similarity error:", e)
        raise HTTPException(status_code=400, detail=str(e))
