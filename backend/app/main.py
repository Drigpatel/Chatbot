from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv

from app.langchain_flow import LangChainFlows
from app.embeddings import EmbeddingIndex

# Load env
load_dotenv()

app = FastAPI(title="ChatBot Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("WARNING: No OPENAI_API_KEY provided!")

flows = LangChainFlows(
    openai_api_key=OPENAI_API_KEY,
    model_name=OPENAI_MODEL
)

QUESTIONS_FILE = BASE_DIR / "questions.json"
EMB = EmbeddingIndex()

try:
    EMB.load()
    print("Embedding index loaded.")
except Exception:
    print("Building embedding index...")
    EMB.build(str(QUESTIONS_FILE))


# ---------------------------
# MODELS
# ---------------------------
class ChatRequest(BaseModel):
    message: str

class RefineRequest(BaseModel):
    question: str
    feedback: str | None = ""


# ---------------------------
# ROUTES
# ---------------------------
@app.get("/")
async def root():
    return {"status": "Chatbot Backend Running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = flows.validate(req.message)
        print("VALIDATION:", result)
        return {"validation": result}
    except Exception as e:
        print("CHAT ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine")
async def refine(req: RefineRequest):
    try:
        result = flows.refine(req.question, req.feedback or "")
        print("REFINE:", result)
        return {"refined_answer": result}
    except Exception as e:
        print("REFINE ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity")
async def similarity(req: RefineRequest):
    try:
        results = EMB.query(req.question, top_k=5)
        similar = [r for r in results if r["score"] >= 0.80]
        return {"similar": similar}
    except Exception as e:
        print("SIMILARITY ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
