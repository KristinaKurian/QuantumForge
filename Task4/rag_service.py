from __future__ import annotations

import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from dotenv import load_dotenv
load_dotenv()
import faiss, httpx, numpy as np
from sentence_transformers import SentenceTransformer
from Task4.security import SECURE_CONTEXT_INSTRUCTION, ProtectionMode, output_guard, protect_chunks, validate_mode

UNKNOWN_ANSWER = "Я не знаю"
FEW_SHOT = """
Example 1
Q: What is the Void Core?
A: The Void Core is a moon-sized Dominion battle station built around a reactor and a weapon capable of destroying an inhabited planet.

Example 2
Q: Who trains Lio Arken on Mirehaven?
A: Eron Kai trains Lio Arken on Mirehaven.
""".strip()
BASE_SYSTEM_PROMPT = """
You are the internal Astraforge knowledge-base assistant.
1. Answer ONLY from supplied CONTEXT. Do not use outside knowledge.
2. If context is insufficient, answer exactly: Я не знаю
3. Do not invent facts.
4. Analyze internally; do not reveal private chain-of-thought.
5. For supported answers provide concise Answer, short Evidence, and Sources.
""".strip()

class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

@dataclass
class OllamaProvider:
    model: str
    base_url: str
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            r=httpx.post(f"{self.base_url.rstrip('/')}/api/chat", timeout=120, json={"model":self.model,"stream":False,"options":{"temperature":0},"messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]})
            r.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}. Start: ollama serve") from exc
        return r.json()["message"]["content"].strip() or UNKNOWN_ANSWER

@dataclass
class OpenAIProvider:
    model: str
    def __post_init__(self):
        from openai import OpenAI
        self.client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        r=self.client.chat.completions.create(model=self.model,temperature=0,messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}])
        return r.choices[0].message.content or UNKNOWN_ANSWER

def build_provider() -> LLMProvider:
    provider=os.getenv("LLM_PROVIDER","ollama").lower()
    if provider=="openai": return OpenAIProvider(os.getenv("OPENAI_MODEL","gpt-5.6-luna"))
    return OllamaProvider(os.getenv("OLLAMA_MODEL","qwen3:8b"), os.getenv("OLLAMA_BASE_URL","http://localhost:11434"))

def system_prompt_for(mode: ProtectionMode) -> str:
    return BASE_SYSTEM_PROMPT + ("\n\n"+SECURE_CONTEXT_INSTRUCTION if mode in {"pre_prompt","all"} else "")

class RAGService:
    def __init__(self,index_dir:Path=Path("Task3/index"),top_k:int|None=None,min_score:float|None=None):
        self.index_dir=index_dir; self.top_k=top_k or int(os.getenv("RAG_TOP_K","4")); self.min_score=min_score if min_score is not None else float(os.getenv("RAG_MIN_SCORE","0.42")); self.default_protection_mode=validate_mode(os.getenv("RAG_PROTECTION_MODE","all"))
        manifest_path=index_dir/"manifest.json"
        if not manifest_path.exists(): raise RuntimeError("FAISS index is missing. Run: python Task3/build_index.py --force")
        self.manifest=json.loads(manifest_path.read_text(encoding="utf-8")); self.index=faiss.read_index(str(index_dir/"faiss.index")); self.chunks=[json.loads(line) for line in (index_dir/"chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.embedding_model=SentenceTransformer(self.manifest["model_name"],device="cpu"); self.llm=build_provider()
    def retrieve(self,question:str)->list[dict]:
        vector=self.embedding_model.encode([question],batch_size=1,convert_to_numpy=True,normalize_embeddings=True,device="cpu"); vector=np.ascontiguousarray(vector,dtype=np.float32); scores,ids=self.index.search(vector,self.top_k)
        out=[]
        for score,idx in zip(scores[0],ids[0]):
            if idx<0: continue
            c=dict(self.chunks[int(idx)]); c["score"]=float(score); out.append(c)
        return out
    def answer(self,question:str,protection_mode:str|None=None)->dict:
        mode=validate_mode(protection_mode or self.default_protection_mode); raw=self.retrieve(question); chunks,events=protect_chunks(raw,mode); safe_top=chunks[0]["score"] if chunks else 0.0
        if not chunks or safe_top<self.min_score:
            return {"answer":UNKNOWN_ANSWER,"top_score":safe_top,"sources":[],"security":{"protection_mode":mode,"events":[e.as_dict() for e in events],"output_blocked":False}}
        context=[]
        for item in chunks:
            label=f"{Path(item['source']).name}#{item['chunk_id']}"; context.append(f"[SOURCE {label} | score={item['score']:.4f}]\n{item['text']}")
        prompt=f"{FEW_SHOT}\n\nCONTEXT:\n"+"\n".join(context)+f"\n\nUSER QUESTION:\n{question}\n\nAnswer from context. If insufficient output only: {UNKNOWN_ANSWER}"
        answer=self.llm.generate(system_prompt_for(mode),prompt).strip()
        if answer.lower().startswith("я не знаю"): answer=UNKNOWN_ANSWER
        blocked=False
        if mode=="all": answer,blocked=output_guard(answer)
        return {"answer":answer,"top_score":safe_top,"sources":[{"source":i["source"],"title":i["title"],"chunk_id":i["chunk_id"],"score":i["score"]} for i in chunks],"security":{"protection_mode":mode,"events":[e.as_dict() for e in events],"output_blocked":blocked}}
