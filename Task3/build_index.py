from __future__ import annotations

# IMPORTANT:
# Configure native libraries BEFORE importing numpy/faiss/torch/transformers.
from safe_runtime import configure_environment

configure_environment()

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from safe_runtime import limit_native_threads

limit_native_threads(faiss)

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"

# The Astraforge documents are intentionally short and entity-focused.
# A 180-word chunk normally keeps one fact group together.
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 30

# Small batches reduce peak RAM and avoid large temporary Torch allocations.
EMBEDDING_BATCH_SIZE = 4


def load_documents(kb_dir: Path) -> list[dict]:
    documents: list[dict] = []

    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        lines = text.splitlines()
        first_line = lines[0].strip() if lines else ""
        title = (
            first_line.lstrip("#").strip()
            if first_line.startswith("#")
            else path.stem
        )

        documents.append(
            {
                "source": path.as_posix(),
                "title": title,
                "text": text,
            }
        )

    return documents


def split_text_by_words(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """
    Deterministic memory-light splitter.

    It stores word positions so every chunk can be traced back to the source.
    The Task 2 documents are already one-entity-per-file, therefore a
    word-based splitter preserves semantic locality well for this corpus.
    """
    words = text.split()

    if not words:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    step = chunk_size - overlap
    chunks: list[dict] = []

    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        chunks.append(
            {
                "chunk_index": chunk_index,
                "word_start": start,
                "word_end": end,
                "word_count": len(chunk_words),
                "text": " ".join(chunk_words),
            }
        )

        if end >= len(words):
            break

        start += step
        chunk_index += 1

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    chunks: list[dict] = []

    for document in documents:
        source_stem = Path(document["source"]).stem
        local_chunks = split_text_by_words(document["text"])

        for item in local_chunks:
            chunks.append(
                {
                    "chunk_id": f"{source_stem}:{item['chunk_index']:03d}",
                    "chunk_index": item["chunk_index"],
                    "source": document["source"],
                    "title": document["title"],
                    "word_start": item["word_start"],
                    "word_end": item["word_end"],
                    "word_count": item["word_count"],
                    "text": item["text"],
                }
            )

    return chunks


def save_chunks(path: Path, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def encode_into_faiss(
    model: SentenceTransformer,
    texts: list[str],
    dimension: int,
) -> faiss.Index:
    """
    Encode in very small batches and add each batch directly to FAISS.

    No multiprocessing pool is created.
    No full N x embedding_dimension matrix is retained in RAM.
    """
    index = faiss.IndexFlatIP(dimension)

    total = len(texts)

    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        end = min(start + EMBEDDING_BATCH_SIZE, total)
        batch = texts[start:end]

        vectors = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )

        # FAISS expects contiguous float32 memory.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != dimension:
            raise RuntimeError(
                f"Unexpected embedding shape: {vectors.shape}; "
                f"expected (*, {dimension})"
            )

        index.add(vectors)

        # Drop temporary Torch/NumPy objects as soon as the batch is indexed.
        del vectors
        del batch
        gc.collect()

        print(f"Encoded {end}/{total} chunks")

    return index


def build_index(kb_dir: Path, index_dir: Path, force: bool = False) -> None:
    index_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.jsonl"
    manifest_path = index_dir / "manifest.json"
    report_path = index_dir / "BUILD_REPORT.md"

    if index_path.exists() and not force:
        print(f"{index_path} already exists.")
        print("Use --force if you really want to rebuild it.")
        return

    started = time.perf_counter()

    documents = load_documents(kb_dir)
    if len(documents) < 30:
        raise SystemExit(
            f"Expected at least 30 knowledge-base documents, got {len(documents)}"
        )

    chunks = build_chunks(documents)
    if not chunks:
        raise SystemExit("No chunks were produced from the knowledge base")

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Loading embedding model on CPU: {MODEL_NAME}")

    # Explicit CPU avoids accidental MPS/Metal use on macOS.
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise RuntimeError("Could not determine embedding dimension")

    dimension = int(dimension)
    print(f"Embedding dimension: {dimension}")

    texts = [chunk["text"] for chunk in chunks]
    index = encode_into_faiss(
        model=model,
        texts=texts,
        dimension=dimension,
    )

    # Model is no longer needed once the FAISS vectors are built.
    del model
    gc.collect()

    index_dir.mkdir(parents=True, exist_ok=True)

    # Write to temporary files first. If the process fails during generation,
    # it will not leave a corrupted final index behind.
    tmp_index_path = index_dir / "faiss.index.tmp"
    faiss.write_index(index, str(tmp_index_path))
    tmp_index_path.replace(index_path)

    save_chunks(chunks_path, chunks)

    duration = time.perf_counter() - started

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "embedding_dimension": dimension,
        "index_type": "faiss.IndexFlatIP",
        "similarity": "cosine via normalized inner product",
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "device": "cpu",
        "multiprocessing": False,
        "duration_seconds": round(duration, 3),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_path.write_text(
        f"""# FAISS index build report

- Model: `{MODEL_NAME}`
- Device: `CPU`
- Embedding dimension: {dimension}
- Documents: {len(documents)}
- Chunks: {len(chunks)}
- Chunk size: {CHUNK_SIZE_WORDS} words
- Chunk overlap: {CHUNK_OVERLAP_WORDS} words
- Embedding batch size: {EMBEDDING_BATCH_SIZE}
- Multiprocessing: disabled
- Native threads: limited to 1
- Index: `FAISS IndexFlatIP`
- Similarity: cosine similarity via normalized vectors
- Build time: {duration:.2f} seconds

The index is built in small batches. Each batch is immediately added to
FAISS, so the full embedding matrix is never retained in memory.
""",
        encoding="utf-8",
    )

    print()
    print("Index successfully built")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("Task2/knowledge_base"),
        help="Directory with Task 2 Markdown documents",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("Task3/index"),
        help="Directory where FAISS index and metadata are stored",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild an existing index",
    )
    args = parser.parse_args()

    build_index(
        kb_dir=args.kb_dir,
        index_dir=args.index_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
