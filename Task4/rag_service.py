from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
load_dotenv()

import faiss
import httpx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from Task4.security import (
    SECURE_CONTEXT_INSTRUCTION,
    ProtectionMode,
    output_guard,
    protect_chunks,
    validate_mode,
)

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

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
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                timeout=120,
                json={
                    "model": self.model,
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.base_url}. "
                "Task 4 does not download Qwen. "
                "Check: curl http://127.0.0.1:11434/api/tags"
            ) from exc
        return response.json()["message"]["content"].strip() or UNKNOWN_ANSWER


@dataclass
class OpenAIProvider:
    model: str

    def __post_init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or UNKNOWN_ANSWER


def build_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        return OpenAIProvider(os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    if provider == "ollama":
        return OllamaProvider(
            os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        )
    raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}")


def system_prompt_for(mode: ProtectionMode) -> str:
    return BASE_SYSTEM_PROMPT + (
        "\n\n" + SECURE_CONTEXT_INSTRUCTION
        if mode in {"pre_prompt", "all"} else ""
    )


def load_embedding_model_offline(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(
            model_name,
            device="cpu",
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Embedding model is not available locally. "
            f"Expected {model_name!r}. Task 4 is offline and will not download it. "
            "Run once: python Task3/build_index.py --force"
        ) from exc


class RAGService:
    def __init__(self, index_dir: Path = Path("Task3/index"),
                 top_k: int | None = None,
                 min_score: float | None = None):
        self.index_dir = index_dir
        self.top_k = top_k if top_k is not None else int(os.getenv("RAG_TOP_K", "4"))
        self.min_score = min_score if min_score is not None else float(os.getenv("RAG_MIN_SCORE", "0.42"))
        self.default_protection_mode = validate_mode(os.getenv("RAG_PROTECTION_MODE", "all"))

        manifest_path = index_dir / "manifest.json"
        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.jsonl"

        missing = [str(p) for p in (manifest_path, index_path, chunks_path) if not p.exists()]
        if missing:
            raise RuntimeError("FAISS index incomplete: " + ", ".join(missing))

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_path))
        self.chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index/chunks mismatch: {self.index.ntotal} != {len(self.chunks)}"
            )

        model_name = self.manifest["model_name"]
        print(f"Loading cached embedding model OFFLINE on CPU: {model_name}")
        self.embedding_model = load_embedding_model_offline(model_name)
        self.llm = build_provider()

    def retrieve(self, question: str) -> list[dict]:
        vector = self.embedding_model.encode(
            [question],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        scores, ids = self.index.search(vector, self.top_k)
        del vector
        gc.collect()

        result = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(score)
            result.append(chunk)
        return result

    def answer(self, question: str, protection_mode: str | None = None) -> dict:
        mode = validate_mode(protection_mode or self.default_protection_mode)
        raw = self.retrieve(question)
        chunks, events = protect_chunks(raw, mode)
        safe_top = chunks[0]["score"] if chunks else 0.0

        if not chunks or safe_top < self.min_score:
            return {
                "answer": UNKNOWN_ANSWER,
                "top_score": safe_top,
                "sources": [],
                "security": {
                    "protection_mode": mode,
                    "events": [e.as_dict() for e in events],
                    "output_blocked": False,
                },
            }

        context = []
        for item in chunks:
            label = f"{Path(item['source']).name}#{item['chunk_id']}"
            context.append(
                f"[SOURCE {label} | score={item['score']:.4f}]\n{item['text']}"
            )

        prompt = (
            f"{FEW_SHOT}\n\nCONTEXT:\n"
            + "\n".join(context)
            + f"\n\nUSER QUESTION:\n{question}\n\n"
            + f"Answer from context. If insufficient output only: {UNKNOWN_ANSWER}"
        )

        answer = self.llm.generate(system_prompt_for(mode), prompt).strip()
        if answer.lower().startswith(UNKNOWN_ANSWER.lower()):
            answer = UNKNOWN_ANSWER

        blocked = False
        if mode == "all":
            answer, blocked = output_guard(answer)

        return {
            "answer": answer,
            "top_score": safe_top,
            "sources": [
                {
                    "source": i["source"],
                    "title": i["title"],
                    "chunk_id": i["chunk_id"],
                    "score": i["score"],
                }
                for i in chunks
            ],
            "security": {
                "protection_mode": mode,
                "events": [e.as_dict() for e in events],
                "output_blocked": blocked,
            },
        }
