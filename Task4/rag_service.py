from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer


UNKNOWN_ANSWER = "Я не знаю"

FEW_SHOT = """
Example 1
Q: What is the Void Core?
A: The Void Core is a moon-sized Dominion battle station built around a reactor and a weapon capable of destroying an inhabited planet.
Evidence:
1. The knowledge-base article describes the Void Core as a Dominion battle station.
2. The same source states that its main weapon can destroy a planet.

Example 2
Q: Who trains Lio Arken on Mirehaven?
A: Eron Kai trains Lio Arken on Mirehaven.
Evidence:
1. The Mirehaven article states that Eron Kai lives there.
2. The Eron Kai article states that he trains Lio in disciplined use of the Aether Weave.
""".strip()

SYSTEM_PROMPT = """
You are the internal Astraforge knowledge-base assistant.

Rules:
1. Answer ONLY from the supplied CONTEXT. Do not use outside knowledge.
2. If the context does not contain enough information, answer exactly: Я не знаю
3. Do not invent names, events, relationships, dates, or technologies.
4. First analyze the context internally. Do NOT reveal private chain-of-thought.
5. For a supported answer, provide a concise answer followed by a short,
   verifiable evidence summary. The evidence summary is not hidden reasoning:
   it should only list facts explicitly present in the supplied sources.
6. Cite sources using the source labels from CONTEXT.
7. Answer in the same language as the user's question when practical.
""".strip()


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class OpenAIProvider:
    model: str

    def __post_init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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


@dataclass
class OllamaProvider:
    model: str
    base_url: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
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
        payload = response.json()
        return payload["message"]["content"].strip() or UNKNOWN_ANSWER


def build_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        return OpenAIProvider(
            model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        )

    if provider == "ollama":
        return OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}")


class RAGService:
    def __init__(
        self,
        index_dir: Path = Path("Task3/index"),
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.top_k = top_k or int(os.getenv("RAG_TOP_K", "4"))
        self.min_score = (
            min_score
            if min_score is not None
            else float(os.getenv("RAG_MIN_SCORE", "0.42"))
        )

        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                "FAISS index is missing. Run: python Task3/build_index.py"
            )

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.chunks = self._load_chunks(index_dir / "chunks.jsonl")
        self.embedding_model = SentenceTransformer(self.manifest["model_name"])
        self.llm = build_provider()

    @staticmethod
    def _load_chunks(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def retrieve(self, question: str) -> list[dict]:
        vector = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, ids = self.index.search(vector, self.top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def answer(self, question: str) -> dict:
        retrieved = self.retrieve(question)
        top_score = retrieved[0]["score"] if retrieved else 0.0

        # Retrieval gate: do not call the LLM when the index has no sufficiently
        # similar evidence. This makes "Я не знаю" deterministic for out-of-domain
        # questions and reduces hallucination risk/cost.
        if not retrieved or top_score < self.min_score:
            return {
                "answer": UNKNOWN_ANSWER,
                "top_score": top_score,
                "sources": [],
                "retrieved": retrieved,
            }

        context_parts = []
        for item in retrieved:
            label = f"{Path(item['source']).name}#{item['chunk_id']}"
            context_parts.append(
                f"[SOURCE {label} | score={item['score']:.4f}]\n{item['text']}"
            )

        user_prompt = f"""
{FEW_SHOT}

CONTEXT:
{chr(10).join(context_parts)}

USER QUESTION:
{question}

Produce:
Answer: <concise answer>
Evidence:
1. <explicit fact from source>
2. <explicit fact from source, if needed>
Sources: <source labels>

If the context is insufficient, output only: {UNKNOWN_ANSWER}
""".strip()

        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt).strip()

        # Second guard. A provider may return extra wording around an unknown answer.
        if answer.lower().startswith("я не знаю"):
            answer = UNKNOWN_ANSWER

        return {
            "answer": answer,
            "top_score": top_score,
            "sources": [
                {
                    "source": item["source"],
                    "title": item["title"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                }
                for item in retrieved
            ],
            "retrieved": retrieved,
        }
