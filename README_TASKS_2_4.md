# Tasks 2–4 — quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python Task2/validate_kb.py
python Task3/build_index.py
python Task3/smoke_test.py

cp .env.example .env
# Configure Ollama or OpenAI in .env.
# OLLAMA_BASE_URL is for local Python; OLLAMA_DOCKER_BASE_URL is for Docker.

uvicorn Task4.app:app --reload
```

Open Swagger:

```text
http://localhost:8000/docs
```

For Docker:

```bash
cp .env.example .env
docker compose up --build
```

Before final submission:

1. Run `Task3/build_index.py` and commit `Task3/index/faiss.index`,
   `chunks.jsonl`, `manifest.json`, and `BUILD_REPORT.md` if repository size allows.
2. Run the five answerable and five unknown queries from `Task4/examples.md`.
3. Save the required 10 screenshots.
4. Calibrate `RAG_MIN_SCORE` using the observed similarity scores.
