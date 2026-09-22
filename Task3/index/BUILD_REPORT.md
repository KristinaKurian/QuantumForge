# FAISS index build report

- Model: `sentence-transformers/multi-qa-mpnet-base-cos-v1`
- Device: `CPU`
- Embedding dimension: 768
- Documents: 39
- Chunks: 39
- Chunk size: 180 words
- Chunk overlap: 30 words
- Embedding batch size: 4
- Multiprocessing: disabled
- Native threads: limited to 1
- Index: `FAISS IndexFlatIP`
- Similarity: cosine similarity via normalized vectors
- Build time: 9.21 seconds

The index is built in small batches. Each batch is immediately added to
FAISS, so the full embedding matrix is never retained in memory.
