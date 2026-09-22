# Task 4 — RAG-бот

## Pipeline

```text
Question
   ↓
Sentence-Transformers
   ↓
FAISS Top-K
   ↓
Relevance threshold
   ├── недостаточно данных → "Я не знаю"
   ↓
Retrieved context
   ↓
Few-shot + system prompt
   ↓
Qwen3-8B через Ollama
   ↓
Answer + evidence + sources
```

Embedding-модель совпадает с Task 3 и загружается **только из локального Hugging Face cache** (`local_files_only=True`). Task 4 не скачивает её повторно.

Qwen запускается отдельно через Ollama; приложение обращается к `http://127.0.0.1:11434/api/chat`.

## Запуск

Подготовьте `.env`:

```bash
cp .env.example .env
```

Проверьте Ollama:

```bash
ollama pull qwen3:8b
curl http://127.0.0.1:11434/api/tags
```

Запустите API без `--reload`:

```bash
python -m uvicorn Task4.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --env-file .env
```

Swagger: `http://127.0.0.1:8000/docs`.

## Endpoints

| Endpoint | Назначение |
|---|---|
| `GET /health` | состояние сервиса |
| `POST /ask` | полный RAG-запрос |
| `GET /search` | диагностика retrieval |

## Поведение

`RAG_MIN_SCORE` задаёт порог релевантности. Если подходящего контекста нет, LLM не должна придумывать ответ и возвращает `Я не знаю`.

Few-shot примеры встроены в prompt. Набор функциональных тестов находится в `Task4/test_cases.json`.
