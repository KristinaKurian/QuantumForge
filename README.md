# QuantumForge — RAG project

Учебный проект по проектированию и реализации RAG-системы для внутренней базы знаний.

## Структура

| Задача | Результат |
|---|---|
| [Task 1](Task1/Project_template.md) | Выбор LLM, embeddings, vector store и инфраструктуры |
| [Task 2](Task2/README.md) | Подготовка обезличенной базы знаний Astraforge |
| [Task 3](Task3/README.md) | Chunking, embeddings и FAISS-индекс |
| [Task 4](Task4/README.md) | FastAPI RAG-бот |
| [Task 5](Task5/README.md) | Prompt-injection тестирование и защита |

## Реализованный стек

| Компонент | Реализация |
|---|---|
| Embeddings | `sentence-transformers/multi-qa-mpnet-base-cos-v1` |
| Vector search | FAISS `IndexFlatIP` |
| LLM | `qwen3:8b` через Ollama |
| API | FastAPI + Uvicorn |
| Security | pre-prompt, filtering/sanitization, output guard |

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cp .env.example .env

python Task2/validate_kb.py
python Task3/build_index.py --force
python Task3/smoke_test.py
```

Убедитесь, что Ollama и модель доступны:

```bash
ollama pull qwen3:8b
curl http://127.0.0.1:11434/api/tags
```

Запустите API:

```bash
python -m uvicorn Task4.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env
```

Swagger: `http://127.0.0.1:8000/docs`.

## Task 5

```bash
python Task5/verify_malicious_index.py
python Task5/run_demo.py --mode all
```

Описание: [Task5/REPORT.md](Task5/REPORT.md).

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Сервис `faiss-index` строит индекс, после чего `rag-bot` запускает API на порту `8000`.
