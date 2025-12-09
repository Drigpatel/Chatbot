from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv

from app.langchain_flow import LangChainFlows
from app.embeddings import EmbeddingIndex
from fastapi.responses import RedirectResponse

# Load .env
load_dotenv()

app = FastAPI(title="ChatBot Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is missing — Chat features may fail.")

flows = LangChainFlows(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)

QUESTIONS_FILE = BASE_DIR / "questions.json"
EMB = EmbeddingIndex()

try:
    EMB.load()
    print("Embedding index loaded successfully.")
except Exception:
    print("No embedding index found, building new one...")
    EMB.build(str(QUESTIONS_FILE))

class ChatRequest(BaseModel):
    message: str

class RefineRequest(BaseModel):
    question: str
    feedback: str | None = ""

@app.get("/")
async def root():
    return {"status": "Chatbot Backend Running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        validation = flows.validate(req.message)
        return {"validation": validation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refine")
async def refine(req: RefineRequest):
    try:
        result = flows.refine(req.question, req.feedback or "")
        return {"refined_answer": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/similarity")
async def similarity(req: RefineRequest):
    try:
        threshold = float(os.getenv("SIM_THRESH", 0.8))
        results = EMB.query(req.question, top_k=5)
        similar = [r for r in results if r["score"] >= threshold]
        return {"similar": similar}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from mangum import Mangum
handler = Mangum(app)