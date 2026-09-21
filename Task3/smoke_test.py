from __future__ import annotations

from safe_runtime import configure_environment

configure_environment()

import gc
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from safe_runtime import limit_native_threads

limit_native_threads(faiss)

INDEX_DIR = Path("Task3/index")

CASES = [
    (
        "Who trained Lio Arken on Mirehaven?",
        "eron_kai.md",
    ),
    (
        "What was the purpose of Directive Blackglass?",
        "directive_blackglass.md",
    ),
    (
        "What weakness was used to destroy the first Void Core?",
        "void_core.md",
    ),
]


def load_chunks() -> list[dict]:
    with (INDEX_DIR / "chunks.jsonl").open(
        encoding="utf-8"
    ) as fh:
        return [
            json.loads(line)
            for line in fh
            if line.strip()
        ]


def main() -> None:
    required = [
        INDEX_DIR / "faiss.index",
        INDEX_DIR / "chunks.jsonl",
        INDEX_DIR / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Index is not built yet. Missing: " + ", ".join(missing)
        )

    manifest = json.loads(
        (INDEX_DIR / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    chunks = load_chunks()
    index = faiss.read_index(
        str(INDEX_DIR / "faiss.index")
    )

    model = SentenceTransformer(
        manifest["model_name"],
        device="cpu",
    )

    failures: list[str] = []

    for query, expected_file in CASES:
        vector = model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )
        vector = np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

        scores, ids = index.search(vector, 3)

        sources = [
            Path(chunks[int(i)]["source"]).name
            for i in ids[0]
            if i >= 0
        ]

        ok = expected_file in sources

        print(f"{'PASS' if ok else 'FAIL'}: {query}")
        print(f"  expected={expected_file}")
        print(f"  top3={sources}")
        print(
            "  scores="
            + str(
                [
                    round(float(score), 4)
                    for score in scores[0]
                ]
            )
        )

        if not ok:
            failures.append(query)

        del vector
        gc.collect()

    del model
    gc.collect()

    if failures:
        raise SystemExit(
            f"{len(failures)} retrieval test(s) failed"
        )

    print()
    print("All retrieval smoke tests passed")


if __name__ == "__main__":
    main()
