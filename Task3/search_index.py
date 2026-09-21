from __future__ import annotations

from safe_runtime import configure_environment

configure_environment()

import argparse
import gc
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from safe_runtime import limit_native_threads

limit_native_threads(faiss)


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [
            json.loads(line)
            for line in fh
            if line.strip()
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("Task3/index"),
    )
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    manifest_path = args.index_dir / "manifest.json"
    index_path = args.index_dir / "faiss.index"
    chunks_path = args.index_dir / "chunks.jsonl"

    if not manifest_path.exists() or not index_path.exists() or not chunks_path.exists():
        raise SystemExit(
            "Index files are missing. Run: python Task3/build_index.py"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    chunks = load_chunks(chunks_path)

    index = faiss.read_index(str(index_path))
    model = SentenceTransformer(
        manifest["model_name"],
        device="cpu",
    )

    vector = model.encode(
        [args.query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cpu",
    )
    vector = np.ascontiguousarray(vector, dtype=np.float32)

    scores, ids = index.search(vector, args.k)

    del vector
    del model
    gc.collect()

    for rank, (score, idx) in enumerate(
        zip(scores[0], ids[0]),
        start=1,
    ):
        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        print()
        print(f"#{rank} score={float(score):.4f}")
        print(f"source={chunk['source']}")
        print(f"title={chunk['title']}")
        print(f"chunk_id={chunk['chunk_id']}")
        print(
            f"position=words "
            f"{chunk['word_start']}:{chunk['word_end']}"
        )
        print(chunk["text"])


if __name__ == "__main__":
    main()
