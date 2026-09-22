from __future__ import annotations
import os
from functools import lru_cache
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from Task4.rag_service import RAGService
app=FastAPI(title="QuantumForge Astraforge RAG Bot",version="1.1.0")
ProtectionModeLiteral=Literal["none","pre_prompt","sanitize","drop","all"]
class AskRequest(BaseModel):
    question:str=Field(min_length=2,max_length=2000)
    protection_mode:ProtectionModeLiteral|None=None
@lru_cache
def get_rag()->RAGService: return RAGService()
@app.get('/health')
def health()->dict:
    rag=get_rag(); return {"status":"ok","model":rag.manifest["model_name"],"chunks":len(rag.chunks),"top_k":rag.top_k,"min_score":rag.min_score,"default_protection_mode":rag.default_protection_mode,"unsafe_demo_enabled":os.getenv("ALLOW_UNSAFE_DEMO","false").lower()=="true"}
@app.post('/ask')
def ask(request:AskRequest)->dict:
    mode=request.protection_mode or get_rag().default_protection_mode
    if mode=="none" and os.getenv("ALLOW_UNSAFE_DEMO","false").lower()!="true": raise HTTPException(status_code=403,detail="Unsafe mode disabled. Set ALLOW_UNSAFE_DEMO=true only for local synthetic demo.")
    return get_rag().answer(request.question,protection_mode=mode)
@app.get('/search')
def search(q:str)->dict: return {"query":q,"results":get_rag().retrieve(q)}
