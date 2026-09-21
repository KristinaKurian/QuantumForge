from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from Task4.rag_service import RAGService

app = FastAPI(
    title="QuantumForge Astraforge RAG Bot",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@lru_cache
def get_rag() -> RAGService:
    return RAGService()


@app.get("/health")
def health() -> dict:
    rag = get_rag()
    return {
        "status": "ok",
        "model": rag.manifest["model_name"],
        "chunks": len(rag.chunks),
        "top_k": rag.top_k,
        "min_score": rag.min_score,
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return get_rag().answer(request.question)


@app.get("/search")
def search(q: str) -> dict:
    rag = get_rag()
    return {
        "query": q,
        "results": rag.retrieve(q),
    }
