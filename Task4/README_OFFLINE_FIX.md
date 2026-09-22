# Task 4 offline fix

Task 4 no longer downloads the embedding model.

Task 3 has already built the FAISS index with:

`sentence-transformers/multi-qa-mpnet-base-cos-v1`

Task 4 now loads that model only from the local Hugging Face cache:

```python
SentenceTransformer(
    model_name,
    device="cpu",
    local_files_only=True,
)
```

It also enables:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
```

Qwen is different: `qwen3:8b` runs in Ollama. Task 4 does not download it.

Check Ollama:

```bash
curl http://127.0.0.1:11434/api/tags
```

Start the API without `--reload` for the demo:

```bash
python -m uvicorn Task4.app:app --host 127.0.0.1 --port 8000 --env-file .env
```

If Task 4 reports that the embedding model is not cached, run once:

```bash
python Task3/build_index.py --force
```

and start Task 4 again.
